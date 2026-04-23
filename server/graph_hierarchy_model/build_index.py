"""
Step 3: Generate Hierarchical Index (Community Summaries)
Input: graph.pkl + chunked conversation JSON
Output: hierarchy.pkl + hierarchy.json
"""

import os
import json
from typing import Dict, Any, List
from collections import defaultdict

import networkx as nx

from server import settings
from utils import call_llm, ensure_dir, load_pickle, save_pickle


def save_json(obj, path):
    """Save object to JSON file"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


COMMUNITY_SUMMARY_PROMPT = settings.prompt_settings.community_summary.get("default")


def load_chunk_texts(chunks_json_path: str) -> Dict[str, str]:
    """
    Load the chunked conversation JSON and return a mapping from chunk_id to concatenated text.
    The concatenated text is a single string formatted from all messages in that chunk.
    """
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    chunk_texts = {}
    for chunk in chunks:
        cid = chunk["chunk_id"]
        messages = chunk.get("messages", [])
        # Format each message as "Role: text"
        lines = []
        for msg in messages:
            role = "User" if msg.get("from") == "user" else "Assistant"
            text = msg.get("text", "")
            if text:
                lines.append(f"{role}: {text}")
        chunk_texts[cid] = "\n".join(lines)
    return chunk_texts


def generate_community_summary(
    community_id: int,
    nodes: List[str],
    G: nx.Graph,
    chunk_texts: Dict[str, str],
    model: str,
) -> str:
    """
    Generate a summary for a single community.
    Nodes: list of node IDs belonging to this community.
    """
    entities = []
    snippets = []

    for node in nodes:
        node_type = G.nodes[node].get("type", "")
        if node_type == "entity":
            name = G.nodes[node].get("name", node)
            entities.append(name)
        elif node_type == "chunk":
            # Retrieve pre-loaded text for this chunk
            text = chunk_texts.get(node, "")
            if text:
                # Take first 400 characters as a snippet to avoid overly long prompts
                snippets.append(text[:400])

    # If very little content, produce a trivial summary
    if len(entities) <= 2 and len(snippets) <= 1:
        return f"Community mainly discusses {', '.join(entities)}."

    entities_str = "; ".join(entities[:30])  # limit number of entities
    snippets_str = "\n---\n".join(snippets[:5])  # limit number of snippets

    prompt = COMMUNITY_SUMMARY_PROMPT.format(
        entities=entities_str,
        text_snippets=snippets_str,
    )
    messages = [{"role": "user", "content": prompt}]
    summary = call_llm(messages, model=model, temperature=0.3)
    return summary.strip()


def build_community_hierarchy(
    G: nx.Graph,
    chunk_texts: Dict[str, str],
    model: str,
) -> Dict[int, Dict[str, Any]]:
    """
    Group nodes by community, generate a summary for each, and return a hierarchy dictionary.
    """
    communities = defaultdict(list)

    # Collect nodes per community
    for node, comm_id in G.nodes(data="community"):
        if comm_id is not None:
            communities[comm_id].append(node)

    if not communities:
        print("No community information found in the graph. Cannot build hierarchy.")
        return {}

    hierarchy = {}
    for comm_id, nodes in communities.items():
        print(f"Generating summary for community {comm_id} ({len(nodes)} nodes)...")
        summary = generate_community_summary(comm_id, nodes, G, chunk_texts, model)

        # Extract entity names and chunk IDs for this community
        community_entities = [
            G.nodes[n].get("name", n)
            for n in nodes
            if G.nodes[n].get("type") == "entity"
        ]
        community_chunks = [
            n for n in nodes if G.nodes[n].get("type") == "chunk"
        ]

        hierarchy[comm_id] = {
            "id": comm_id,
            "summary": summary,
            "node_count": len(nodes),
            "entities": community_entities,
            "chunks": community_chunks,
        }

    return hierarchy


def main():
    # Use settings directly instead of loading config
    output_dir = str(settings.basic_settings.OUTPUT_PATH)
    ensure_dir(output_dir)

    # Load graph
    graph_path = os.path.join(output_dir, "graph.pkl")
    if not os.path.exists(graph_path):
        print(f"Error: graph.pkl not found at {graph_path}. Run build_graph.py first.")
        return
    G = load_pickle(graph_path)

    # Load chunked conversation JSON (adjust filename if needed)
    chunks_json_path = os.path.join(output_dir, "chunked_conversation.json")
    if not os.path.exists(chunks_json_path):
        print(f"Error: chunked conversation JSON not found at {chunks_json_path}.")
        return
    chunk_texts = load_chunk_texts(chunks_json_path)

    # Use the model from settings
    model = settings.api_model_settings.SCENARIO_MODELS.get("llm", "glm-4-plus")
    print(f"Using model: {model}")

    hierarchy = build_community_hierarchy(G, chunk_texts, model)

    if not hierarchy:
        print("No communities to save.")
        return

    # Save as pickle for downstream scripts
    hierarchy_pkl_path = os.path.join(output_dir, "hierarchy.pkl")
    save_pickle(hierarchy, hierarchy_pkl_path)

    # Also save as JSON for human inspection
    hierarchy_json_path = os.path.join(output_dir, "hierarchy.json")
    save_json(hierarchy, hierarchy_json_path)

    print(f"Hierarchy saved: {len(hierarchy)} communities → {hierarchy_pkl_path} and {hierarchy_json_path}")


if __name__ == "__main__":

    main()