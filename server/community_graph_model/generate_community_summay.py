"""
Step 3: Generate Hierarchical Index (Community Summaries)
Input: community_graph.pkl + chunked conversation JSON
Output: summary.pkl + summary.json
"""

import os
import json
import pickle
from typing import Dict, Any, List
from collections import defaultdict

import networkx as nx

from server import settings

COMMUNITY_SUMMARY_PROMPT = settings.prompt_settings.community_summary.get("default")


def load_chunk_texts(chunks_json_path: str) -> Dict[str, str]:
    """
    Load the chunked conversation JSON and return a mapping from chunk_id to concatenated text.
    """
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = data.get("chunks", [])
    chunk_texts = {}
    for chunk in chunks:
        cid = chunk["chunk_id"]
        messages = chunk.get("messages", [])
        lines = []
        for msg in messages:
            sender = msg.get("from", "unknown")
            text = msg.get("text", "")
            if text:
                lines.append(f"{sender}: {text}")
        chunk_texts[cid] = "\n".join(lines)
    return chunk_texts


def generate_a_summary(
    nodes: List[str],
    G: nx.Graph,
    chunk_texts: Dict[str, str],
    model: str,
    all_source_ids: set
) -> str:

    # 1. Prepare Entities + Relations + Snippets As Prompt
    entities = []
    relations = set()
    snippets = []
    for node in nodes:
        entities.append(G.nodes[node].get("name", node))
        # Search For Relation Descriptions Between The Entity And Its Neighbors
        for neighbor in G.neighbors(node):
            if neighbor in nodes:
                edge_data = G.edges[node, neighbor]
                rel_desc = edge_data.get("relation", "")
                if rel_desc:
                    src_name = G.nodes[node].get("name", node)
                    tgt_name = G.nodes[neighbor].get("name", neighbor)
                    if src_name > tgt_name:
                        src_name, tgt_name = tgt_name, src_name
                    relations.add(f"{src_name} -> {tgt_name}: {rel_desc}")
    snippets = [chunk_texts.get(cid, "") for cid in all_source_ids if cid in chunk_texts]

    # In Case That There Are Too Many Entities, Snippets, Or Relations
    entities_str = "; ".join(entities[:30])
    snippets_str = "\n---\n".join(snippets[:5])
    relations_str = "; ".join(list(relations)[:10])
    prompt = COMMUNITY_SUMMARY_PROMPT.format(
        entities=entities_str,
        text_snippets=snippets_str,
        relations=relations_str if relations_str else "None",
    )
    messages = [{"role": "user", "content": prompt}]

    # 2. Call LLM API
    from langchain_openai import ChatOpenAI
    platform_config = settings.api_model_settings.MODEL_PLATFORMS.get("deepseek")
    model_config = settings.api_model_settings.MODELS.get(model)
    llm = ChatOpenAI(
        model=model,
        api_key=platform_config.api_key if platform_config else None,
        base_url=platform_config.base_url if platform_config else None,
        temperature=model_config.temperature if model_config else None,
    )
    response = llm.invoke(messages)
    return response.content.strip()


def build_summaries(
    G: nx.Graph,
    chunk_texts: Dict[str, str],
    model: str,
) -> Dict[int, Dict[str, Any]]:
    """
    Group nodes by community, generate a summary for each, and return a hierarchy dictionary.
    """
    # 1. Prepare Dict(Community_id -> Nodes)
    communities = defaultdict(list)
    for node, attr in G.nodes(data=True):
        comm_id = attr.get("community")
        if comm_id is not None:
            communities[comm_id].append(node)
    if not communities:
        print("No community information found in the graph. Cannot build hierarchy.")
        return {}

    # 2. Generate Summaries
    summaries = {}
    for comm_id, nodes in communities.items():
        print(f"Generating summary for community {comm_id} ({len(nodes)} nodes)...")

        # Source Chunk Ids
        all_source_ids = set()
        for n in nodes:
            all_source_ids.update(G.nodes[n].get("source_ids", []))

        # Build A Summary
        summary = generate_a_summary(nodes, G, chunk_texts, model, all_source_ids)

        # Output
        summaries[comm_id] = {
            "community_id": comm_id,
            "summary": summary,
            "entity_count": len(nodes),
            "entities": [G.nodes[n].get("name") for n in nodes],
            "source_chunk_ids": list(all_source_ids),
        }
    return summaries


def generate_summaries(graph_pkl_path: str, json_path: str, output_dir: str) -> str:
    # 1. Load graph
    if not os.path.exists(graph_pkl_path):
        print(f"Error: community_graph.pkl not found at {graph_pkl_path}. Run build_graph.py first.")
        return ""
    with open(graph_pkl_path, "rb") as f:
        G = pickle.load(f)

    # 2. Load chunked conversation JSON
    if not os.path.exists(json_path):
        print(f"Error: chunked conversation JSON not found at {json_path}.")
        return ""
    chunk_texts = load_chunk_texts(json_path)

    # 3. Build Community Summary
    model = settings.api_model_settings.DEFAULT_SUMMARY_MODEL
    print(f"Using LLM model: {model}")
    summary = build_summaries(G, chunk_texts, model)
    if not summary:
        print("No communities to save.")
        return ""

    # 4. Save to PKL and JSON
    report = {
        "from_graph": graph_pkl_path,
        "summaries": summary,
    }
    summary_pkl_path = os.path.join(output_dir, "summary.pkl")
    with open(summary_pkl_path, 'wb') as f:
        pickle.dump(report, f, protocol=4)
    summary_json_path = os.path.join(output_dir, "summary.json")
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print(f"Hierarchy saved: {len(summary)} communities → {summary_pkl_path} and {summary_json_path}")
    return summary_pkl_path


if __name__ == "__main__":
    output_dir = "D:\\MyProjects\\ragIM\\data\\outputs"
    graph_pkl_path = os.path.join(output_dir, "community_graph_1.0.pkl")
    json_path = "D:\\MyProjects\\ragIM\\data\\processed_chunks\\ibm_graph_hierarchy_split.json"
    generate_summaries(graph_pkl_path, json_path, output_dir)