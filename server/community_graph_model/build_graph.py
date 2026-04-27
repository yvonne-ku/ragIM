import os
import json
import pickle

import networkx as nx
from typing import List, Dict, Any, Tuple

from server import settings

EXTRACT_ENTITY_PROMPT = settings.prompt_settings.entity_extraction.get("default")

from networkx.readwrite import json_graph


def save_graph_to_json(G: nx.Graph, file_path: str):
    """
    将 NetworkX 图转换为 JSON 兼容格式并保存
    """
    data = json_graph.node_link_data(G)
    for node in data.get("nodes", []):
        if "source_ids" in node and isinstance(node["source_ids"], set):
            node["source_ids"] = list(node["source_ids"])
    for edge in data.get("links", []):  # node_link_data 默认将边称为 links
        if "source_ids" in edge and isinstance(edge["source_ids"], set):
            edge["source_ids"] = list(edge["source_ids"])
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


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
    # 1. Prepare prompt
    conv_text = format_conversation_text(messages)
    prompt = EXTRACT_ENTITY_PROMPT.format(conversation=conv_text)
    llm_messages = [{"role": "user", "content": prompt}]

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
    response = llm.invoke(llm_messages)
    response_text = response.content  # 提取文本字符串

    # 3. Parse JSON from response
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No valid JSON found")
        json_str = response_text[start:end]
        data = json.loads(json_str)
        entities = data.get("entities", [])
        relations = data.get("relations", [])
        return entities, relations
    except Exception as e:
        print(f"Failed to parse LLM output: {e}\nRaw output: {response_text[:200]}...")
        return [], []


def build_graph_from_chunks(chunks_data: List[Dict[str, Any]], model: str) -> nx.Graph:
    """
    Build a global graph from all chunks.
    """
    G = nx.Graph()
    for chunk in chunks_data:
        chunk_id = chunk["chunk_id"]
        messages = chunk["messages"]

        # 1. Extract Entities and Relations from Each Chunk
        entities, relations = extract_entities_relations_from_chunk(messages, model)

        # 2. Build Entity Nodes
        for ent_name in entities:
            ent_id = f"entity::{ent_name}"
            if ent_id not in G:
                G.add_node(ent_id, type="entity", name=ent_name, source_ids={chunk_id})
            else:
                G.nodes[ent_id]["source_ids"].add(chunk_id)

        # 3. Build Edges
        for rel in relations:
            src_id = f"entity::{rel.get('source')}"
            tgt_id = f"entity::{rel.get('target')}"
            desc = rel.get("description", "")

            if src_id not in G or tgt_id not in G: continue

            if G.has_edge(src_id, tgt_id):
                edge_data = G.edges[src_id, tgt_id]
                edge_data["source_ids"].add(chunk_id)
                if desc and desc not in edge_data["relation"]:
                    edge_data["relation"] += f"；{desc}"
            else:
                G.add_edge(src_id, tgt_id, relation=desc, source_ids={chunk_id})
    return G


def detect_communities_hierarchical(G: nx.Graph, resolution_parameter: float = 1.0):
    """
    Detect just one community in the graph.
    """
    import igraph as ig
    import leidenalg

    # 1. Convert NetworkX graph to igraph
    nodes = list(G.nodes())
    mapping = {node: i for i, node in enumerate(nodes)}
    rev_mapping = {i: node for node, i in mapping.items()}
    ig_graph = ig.Graph()
    ig_graph.add_vertices(len(nodes))
    ig_graph.add_edges([(mapping[u], mapping[v]) for u, v in G.edges()])

    # 2. Run Leiden once
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution_parameter
    )

    # 3. Write community IDs back to NetworkX graph
    for i, comm_id in enumerate(partition.membership):
        node_name = rev_mapping[i]
        G.nodes[node_name]['community'] = comm_id
    return G


def build_graph(json_path: str, output_dir: str, resolution_parameter: float = 1.0, rebuild_graph: bool = True) -> str:
    if not os.path.exists(json_path):
        print(f"Error: chunked file {json_path} not found. Run chunking first.")
        return ""
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    chunks_list = chunks_data["chunks"]
    print(f"Loaded {len(chunks_list)} chunks.")

    # 1. Build Graph Or Get Graph From Existing PKL File
    if rebuild_graph:
        model = settings.api_model_settings.DEFAULT_EXTRACT_ENTITY_MODEL
        print(f"Using LLM model: {model}")
        G = build_graph_from_chunks(chunks_list, model)

        graph_path = os.path.join(output_dir, "raw_graph.pkl")
        with open(graph_path, 'wb') as f:
            pickle.dump(G, f, protocol=4)
        print(f"Graph saved to {graph_path}")
    else:
        graph_path = os.path.join(output_dir, "raw_graph.pkl")
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)
        print(f"Graph loaded from {graph_path}")

    # 2. Community detection By Leiden Algorithm
    method = "leiden"
    print(f"Running {method} community detection...")
    G = detect_communities_hierarchical(G, resolution_parameter)
    if G:
        communities = nx.get_node_attributes(G, 'community').values()
        print(f"Found {len(set(communities))} communities.")
    else:
        print("Community detection failed, graph will not have community info.")

    # 3. Save graph
    # 3.1 As PKL
    pkl_path = os.path.join(output_dir, f"community_graph_{resolution_parameter}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(G, f, protocol=4)
    print(f"Graph (PKL) saved to {pkl_path}")
    # 3.2 As JSON（测性能的时候记得注释掉）
    json_output_path = os.path.join(output_dir, f"community_graph_{resolution_parameter}.json")
    save_graph_to_json(G, json_output_path)
    print(f"Graph (JSON) saved to {json_output_path}")
    return pkl_path


if __name__ == "__main__":
    json_path = "D:\\MyProjects\\ragIM\\data\\processed_chunks\\ibm_graph_hierarchy_split.json"
    output_dir = "D:\\MyProjects\\ragIM\\data\\outputs"
    build_graph(json_path, output_dir, resolution_parameter=1.0, rebuild_graph=True)