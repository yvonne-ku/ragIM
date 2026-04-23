"""
Step 2: Build Entity-Relation Graph (English conversation, GPT-4o-mini)
Input: chunked JSON file (e.g., chunked_conversation.json)
Output: graph.pkl (NetworkX graph) saved in output directory
"""

import os
import json
import pickle

import networkx as nx
from typing import List, Dict, Any, Tuple

from server import settings

EXTRACT_ENTITY_PROMPT = settings.prompt_settings.entity_extraction.get("default")


def format_conversation_text(messages: List[Dict[str, Any]]) -> str:
    """
    Convert a list of message dicts to a plain English dialogue string.
    """
    lines = []
    for msg in messages:
        sender = msg.get("from", "unknown")
        text = msg.get("text", "")
        if text:
            lines.append(f"{sender}: {text}")
    return "\n".join(lines)


def extract_entities_relations_from_chunk(
    messages: List[Dict[str, Any]],
    model: str
) -> Tuple[List[str], List[Dict]]:
    """
    Call LLM to extract entities and relations from one chunk.
    """
    # Prepare prompt
    conv_text = format_conversation_text(messages)
    prompt = EXTRACT_ENTITY_PROMPT.format(conversation=conv_text)
    llm_messages = [{"role": "user", "content": prompt}]

    # Call LLM API
    from langchain_openai import ChatOpenAI
    platform_config = settings.api_model_settings.MODEL_PLATFORMS.get("openai")
    model_config = settings.api_model_settings.MODELS.get(model)
    llm = ChatOpenAI(
            model=model,
            api_key=platform_config.api_key if platform_config else None,
            base_url=platform_config.api_llm_base_url if platform_config else None,
            temperature=model_config.temperature if model_config else None,
        )
    response = llm.invoke(llm_messages)

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
    Heterogeneous Graph: Have two kinds of nodes: chunk nodes and entity nodes.
    """
    G = nx.Graph()

    # Build Chunk Nodes
    for chunk in chunks_data:
        chunk_id = chunk["chunk_id"]
        G.add_node(chunk_id, type="chunk", chunk_id=chunk_id)

    for chunk in chunks_data:
        chunk_id = chunk["chunk_id"]
        messages = chunk["messages"]
        print(f"Processing {chunk_id} ({len(messages)} messages)...")

        entities, relations = extract_entities_relations_from_chunk(messages, model)

        # Build Entity Nodes And Connect To Chunk Nodes
        for ent_name in entities:
            ent_id = f"entity::{ent_name}"
            if ent_id not in G:
                G.add_node(ent_id, type="entity", name=ent_name)
            G.add_edge(ent_id, chunk_id, relation="mentioned_in")

        # Build Relation Edges
        for rel in relations:
            src_name = rel.get("source")
            tgt_name = rel.get("target")
            desc = rel.get("description", "")

            if not src_name or not tgt_name:
                continue

            src_id = f"entity::{src_name}"
            tgt_id = f"entity::{tgt_name}"
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
    return a dict like:
        {
            "chunk_00001": 0,
            "chunk_00002": 0,
            "entity::savings account": 0,
            "entity::freelancer": 1,
            "entity::business account": 1,
            "chunk_00003": 1,
            ...
        }
    the value is the index of the community.
    """
    if method == "leiden":
        try:
            import igraph as ig
            import leidenalg

            # String nodes to integers for Leiden algorithm, chunk_00001 -> 0, entity::entity1 -> 1
            mapping = {node: i for i, node in enumerate(G.nodes())}
            rev_mapping = {i: node for node, i in mapping.items()}

            # Convert to igraph graph
            ig_graph = ig.Graph()
            ig_graph.add_vertices(len(mapping))
            ig_graph.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])

            # Run Leiden algorithm
            partition = leidenalg.find_partition(
                ig_graph, leidenalg.ModularityVertexPartition
            )

            # Map back to original node names, 0 -> chunk_00001, 1 -> entity::entity1
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


def main(json_file_path: str, output_dir: str):
    if not os.path.exists(json_file_path):
        print(f"Error: chunked file {json_file_path} not found. Run chunking first.")
        return
    with open(json_file_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    chunks_list = chunks_data["chunks"]
    print(f"Loaded {len(chunks_list)} chunks.")

    # 1. Build Graph
    model = settings.api_model_settings.DEFAULT_EXTRACT_ENTITY_MODEL
    print(f"Using LLM model: {model}")
    G = build_graph_from_chunks(chunks_list, model)

    # 2. Community detection By Leiden Algorithm
    method = "leiden"
    print(f"Running {method} community detection...")
    node_communities = detect_communities(G, method)
    if node_communities:
        nx.set_node_attributes(G, node_communities, "community")
        print(f"Found {len(set(node_communities.values()))} communities.")
    else:
        print("Community detection failed, graph will not have community info.")

    # 3. Save graph
    graph_path = os.path.join(output_dir, "graph.pkl")
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f, protocol=4)
    print(f"Graph saved to {graph_path}")


if __name__ == "__main__":
    json_file_path = "D:\\MyProjects\\ragIM\\data\\processed_chunks\\ibm_graph_hierarchy_split.json"
    output_dir = "D:\\MyProjects\\ragIM\\data\\outputs"
    main(json_file_path, output_dir)
