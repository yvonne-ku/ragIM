"""
Step 3: Generate Hierarchical Index (Community Summaries)
Input: community_graph.pkl + chunked conversation JSON
Output: hierarchy.pkl + hierarchy.json
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
            sender = msg.get("from", "unknown")
            text = msg.get("text", "")
            if text:
                lines.append(f"{sender}: {text}")
        chunk_texts[cid] = "\n".join(lines)
    return chunk_texts


def generate_community_summary(
    community_id: int,
    nodes: List[str],
    G: nx.Graph,
    chunk_texts: Dict[str, str],
    model: str,
) -> str:

    # 1. Prepare Prompt: Entities + Descriptions + Chunk Texts
    entities = []
    snippets = []
    relations = set()
    for node in nodes:
        node_type = G.nodes[node].get("type", "")
        if node_type == "entity":
            name = G.nodes[node].get("name", node)
            entities.append(name)
            # Search For Relation Descriptions Between The Entity And Its Neighbors In The Same Community
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor].get("type") == "entity" and neighbor in nodes:
                    edge_data = G.edges[node, neighbor]
                    rel_desc = edge_data.get("relation", "")
                    if rel_desc:
                        # To Prevent Duplicate Relations, Sort By Alphabetical Order To Form A Single Record
                        src_name = name
                        tgt_name = G.nodes[neighbor].get("name", neighbor)
                        if src_name > tgt_name:
                            src_name, tgt_name = tgt_name, src_name
                        relations.add(f"{src_name} -> {tgt_name}: {rel_desc}")
        elif node_type == "chunk":
            text = chunk_texts.get(node, "")
            if text:
                snippets.append(text)
    # Generate Short Summary If Not Enough Data
    if len(entities) <= 2 and len(snippets) <= 1:
        return f"Community mainly discusses {', '.join(entities)}."

    # In Case That There Are Too Many Entities, Snippets, Or Relations, Limit Them To 30, 5, 10 Respectively
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
    platform_config = settings.api_model_settings.MODEL_PLATFORMS.get("openai")
    model_config = settings.api_model_settings.MODELS.get(model)
    llm = ChatOpenAI(
        model=model,
        api_key=platform_config.api_key if platform_config else None,
        base_url=platform_config.api_llm_base_url if platform_config else None,
        temperature=model_config.temperature if model_config else None,
    )
    response = llm.invoke(messages)
    return response.content.strip()

"""
 - 生成社区只用前 30 个实体，前 5 个 chunk，前 10 个关系
"""
def build_community_hierarchy(
    G: nx.Graph,
    chunk_texts: Dict[str, str],
    model: str,
) -> Dict[int, Dict[str, Any]]:
    """
    Group nodes by community, generate a summary for each, and return a hierarchy dictionary.
    the return is like:
    {
        0: {
            "id": 0,
            "summary": "This community focuses on business banking for freelancers, including recommendations "
                       "to separate personal and business finances, use of current accounts, and the benefits "
                       "of opening additional free checking or savings accounts.",
            "node_count": 12,
            "entities": [
                "freelancer",
                "business account",
                "savings account",
                "current account",
                "personal finance",
                "checking account",
                "IRA",
                "discount brokerage"
            ],
            "chunks": [
                "chunk_00001",
                "chunk_00003",
                "chunk_00007"
            ]
        },
        1: {
            "id": 1,
            "summary": "This community discusses tax-related concerns for independent contractors...",
            "node_count": 8,
            "entities": ["tax deduction", "1099 form", "independent contractor"],
            "chunks": ["chunk_00002", "chunk_00005"]
        },
        ...
    }
    """
    # 1. Collect nodes per community
    communities = defaultdict(list)
    for node, comm_id in G.nodes(data="community"):
        if comm_id is not None:
            communities[comm_id].append(node)
    if not communities:
        print("No community information found in the graph. Cannot build hierarchy.")
        return {}

    # 2. Generate summaries for each community
    hierarchy = {}
    for comm_id, nodes in communities.items():
        print(f"Generating summary for community {comm_id} ({len(nodes)} nodes)...")
        summary = generate_community_summary(comm_id, nodes, G, chunk_texts, model)

        # Extract entity names and chunk IDs for this community
        community_entities = [
            G.nodes[n].get("name", n)
            for n in nodes if G.nodes[n].get("type") == "entity"
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


def main(graph_pkl_path: str, json_file_path: str, output_dir: str):
    # 1. Load graph
    if not os.path.exists(graph_pkl_path):
        print(f"Error: community_graph.pkl not found at {graph_pkl_path}. Run build_graph.py first.")
        return
    with open(graph_pkl_path, "rb") as f:
        G = pickle.load(f)

    # 2. Load chunked conversation JSON (adjust filename if needed)
    if not os.path.exists(json_file_path):
        print(f"Error: chunked conversation JSON not found at {json_file_path}.")
        return
    chunk_texts = load_chunk_texts(json_file_path)

    # 3. Build Community Hierarchy Summary
    model = settings.api_model_settings.DEFAULT_SUMMARY_MODEL
    print(f"Using LLM model: {model}")
    hierarchy = build_community_hierarchy(G, chunk_texts, model)
    if not hierarchy:
        print("No communities to save.")
        return

    # 4. Save as pickle for downstream scripts
    hierarchy_pkl_path = os.path.join(output_dir, "hierarchy.pkl")
    with open(hierarchy_pkl_path, 'wb') as f:
        pickle.dump(hierarchy, f, protocol=4)

    # 5. Save as JSON for human inspection
    hierarchy_json_path = os.path.join(output_dir, "hierarchy.json")
    with open(hierarchy_json_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=4)

    print(f"Hierarchy saved: {len(hierarchy)} communities → {hierarchy_pkl_path} and {hierarchy_json_path}")


if __name__ == "__main__":
    graph_pkl_path = "/data/outputs/community_graph.pkl"
    json_file_path = "D:\\MyProjects\\ragIM\\data\\processed_chunks\\ibm_graph_hierarchy_split.json"
    output_dir = "D:\\MyProjects\\ragIM\\data\\outputs"
    main(graph_pkl_path, json_file_path, output_dir)
