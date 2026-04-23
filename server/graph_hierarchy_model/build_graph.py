"""
Step 2: Build Entity-Relation Graph (English conversation, GPT-4o-mini)
Input: chunked JSON file (e.g., chunked_conversation.json)
Output: graph.pkl (NetworkX graph) saved in output directory
"""

import os
import json
import networkx as nx
from typing import List, Dict, Any, Tuple

from server import settings
from utils import (
    load_config,
    setup_openai,
    call_llm,
    ensure_dir,
    save_pickle,
    load_json,
)


EXTRACT_ENTITY_PROMPT = settings.prompt_settings.entity_extraction.get("default")


def format_conversation_text(messages: List[Dict[str, Any]]) -> str:
    """
    Convert a list of message dicts to a plain English dialogue string.
    """
    lines = []
    for msg in messages:
        role = "User" if msg.get("from") == "user" else "Assistant"
        text = msg.get("text", "")
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def extract_entities_relations_from_chunk(
    messages: List[Dict[str, Any]],
    model: str
) -> Tuple[List[str], List[Dict]]:
    """
    Call LLM to extract entities and relations from one chunk.
    """
    conv_text = format_conversation_text(messages)

    # Skip very short snippets to save API calls
    if len(conv_text.strip()) < 20:
        return [], []

    prompt = EXTRACT_ENTITY_PROMPT.format(conversation=conv_text)
    llm_messages = [{"role": "user", "content": prompt}]
    response = call_llm(llm_messages, model=model, temperature=0.0)

    # Parse JSON from response
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No valid JSON found")
        json_str = response[start:end]
        data = json.loads(json_str)
        entities = data.get("entities", [])
        relations = data.get("relations", [])
        return entities, relations
    except Exception as e:
        print(f"Failed to parse LLM output: {e}\nRaw output: {response[:200]}...")
        return [], []


def build_graph_from_chunks(chunks_data: List[Dict[str, Any]], model: str) -> nx.Graph:
    """
    Build a global graph from all chunks.
    """
    G = nx.Graph()

    for chunk in chunks_data:
        chunk_id = chunk["chunk_id"]
        G.add_node(chunk_id, type="chunk", chunk_id=chunk_id)

    for chunk in chunks_data:
        chunk_id = chunk["chunk_id"]
        messages = chunk["messages"]
        print(f"Processing {chunk_id} ({len(messages)} messages)...")

        entities, relations = extract_entities_relations_from_chunk(messages, model)

        # Add entity nodes and connect to chunk
        for ent_name in entities:
            ent_id = f"entity::{ent_name}"
            if ent_id not in G:
                G.add_node(ent_id, type="entity", name=ent_name)
            G.add_edge(ent_id, chunk_id, relation="mentioned_in")

        # Add relation edges
        for rel in relations:
            src_name = rel.get("source")
            tgt_name = rel.get("target")
            desc = rel.get("description", "")

            if not src_name or not tgt_name:
                continue

            src_id = f"entity::{src_name}"
            tgt_id = f"entity::{tgt_name}"

            # Ensure nodes exist
            if src_id not in G:
                G.add_node(src_id, type="entity", name=src_name)
            if tgt_id not in G:
                G.add_node(tgt_id, type="entity", name=tgt_name)

            # Merge relation descriptions if edge already exists
            if G.has_edge(src_id, tgt_id):
                existing_desc = G.edges[src_id, tgt_id].get("relation", "")
                if desc not in existing_desc:
                    new_desc = f"{existing_desc}；{desc}" if existing_desc else desc
                    G.edges[src_id, tgt_id]["relation"] = new_desc
            else:
                G.add_edge(src_id, tgt_id, relation=desc)

    print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def detect_communities(G: nx.Graph, method: str = "leiden") -> Dict[str, int]:
    """
    Community detection. Falls back to Louvain if Leiden is not available.
    """
    if method == "leiden":
        try:
            import igraph as ig
            import leidenalg

            mapping = {node: i for i, node in enumerate(G.nodes())}
            rev_mapping = {i: node for node, i in mapping.items()}
            ig_graph = ig.Graph()
            ig_graph.add_vertices(len(mapping))
            ig_graph.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])

            partition = leidenalg.find_partition(
                ig_graph, leidenalg.ModularityVertexPartition
            )
            communities = {
                rev_mapping[i]: comm
                for i, comm in enumerate(partition.membership)
            }
            return communities
        except ImportError:
            print("Leiden not installed, falling back to Louvain.")
            method = "louvain"

    if method == "louvain":
        try:
            import community as community_louvain
            return community_louvain.best_partition(G)
        except ImportError:
            print("python-louvain not installed. Cannot perform community detection.")
            return {}

    return {}


def main():
    config = load_config()
    setup_openai(config)

    output_dir = config["paths"]["output_dir"]
    ensure_dir(output_dir)

    # --- Modify this if your chunked file name differs ---
    input_json = os.path.join(output_dir, "chunked_conversation.json")
    if not os.path.exists(input_json):
        print(f"Error: chunked file {input_json} not found. Run chunking first.")
        return

    chunks_data = load_json(input_json)
    if "chunks" not in chunks_data:
        print("Error: input JSON missing 'chunks' key.")
        return

    chunks_list = chunks_data["chunks"]
    print(f"Loaded {len(chunks_list)} chunks.")

    # Use the model from settings
    model = settings.api_model_settings.SCENARIO_MODELS.get("extract_entity", "GPT-4o-mini")
    print(f"Using LLM model: {model}")

    G = build_graph_from_chunks(chunks_list, model)

    # Community detection
    method = config["graph"]["community_detection"]
    print(f"Running {method} community detection...")
    node_communities = detect_communities(G, method)
    if node_communities:
        nx.set_node_attributes(G, node_communities, "community")
        print(f"Found {len(set(node_communities.values()))} communities.")
    else:
        print("Community detection failed, graph will not have community info.")

    # Save graph
    graph_path = os.path.join(output_dir, "graph.pkl")
    save_pickle(G, graph_path)
    print(f"Graph saved to {graph_path}")


if __name__ == "__main__":
    main()