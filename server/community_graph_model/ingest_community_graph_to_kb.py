"""
Step 4: Ingest Community Summaries And Graph Entities into Separate Vector Knowledge Base
Input: summary.pkl + graph.pkl
Output: ChromaDB Vector Knowledge Base
"""
import json
import os
import pickle
from typing import Dict, Any, List

import networkx as nx
from langchain_core.documents import Document
from server.kb_singleton_util import get_kb


def process_summaries_to_docs(summaries: Dict[int, Dict[str, Any]]) -> List[Document]:
    """
    Convert community summaries to LangChain Document objects
    """
    documents = []
    for comm_id, summary_data in summaries.items():
        # Create document content with summary and related entities
        content = f"{summary_data.get('summary', '')}\n\n"
        content += f"Entities: {', '.join(summary_data.get('entities', []))}\n"

        source_chunks_list = summary_data.get('source_chunks', [])
        source_chunks_str = json.dumps(source_chunks_list)

        # Create metadata
        metadata = {
            "community_id": comm_id,
            "entity_count": summary_data.get('entity_count', 0),
            "source_chunks": source_chunks_str,
        }

        # Create Document object
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    return documents


def ingest_summaries_to_kb(
        summary_pkl_path: str,
        kb_name_summary: str,
    ):
    """
    Ingest community summaries into ChromaDB vector knowledge base
    """
    # 1. Load summary data
    if not os.path.exists(summary_pkl_path):
        print(f"Error: summary.pkl not found at {summary_pkl_path}.")
        return
    with open(summary_pkl_path, "rb") as f:
        report = pickle.load(f)
    summaries = report.get("summaries", {})
    if not summaries:
        print("No summary data found. Cannot ingest into KB.")
        return

    # 2. Convert summaries to documents
    documents = process_summaries_to_docs(summaries)
    print(f"Converted {len(documents)} community summaries to documents.")

    # 3. Get or create knowledge base
    kb = get_kb(kb_name=kb_name_summary)

    # 4. Clear existing collection (optional)
    print(f"Clearing existing collection: {kb_name_summary}")
    kb.delete_collection()

    # 5. Add documents to KB
    print(f"Adding {len(documents)} documents to KB...")
    ids = kb.add_documents(documents)
    print(f"Successfully added {len(ids)} documents to KB.")

    return


def process_entities_to_docs(G: nx.Graph) -> List[Document]:
    documents = []
    for node_id, attrs in G.nodes(data=True):
        entity_name = attrs.get("name", node_id)
        metadata = {
            "entity_id": node_id,
            "entity_name": entity_name,
            "source_chunks": json.dumps(list(attrs.get("source_ids", []))),
        }
        content = entity_name
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    return documents


def ingest_entities_to_kb(
        graph_pkl_path: str,
        kb_name_entity: str,
):
    """
    Ingest community entities into ChromaDB vector knowledge base
    """
    # 1. Load summary data
    if not os.path.exists(graph_pkl_path):
        print(f"Error: graph.pkl not found at {graph_pkl_path}.")
        return
    with open(graph_pkl_path, "rb") as f:
        G = pickle.load(f)
    if not G:
        print("No graph data found. Cannot ingest into KB.")
        return

    # 2. Convert entities to documents
    documents = process_entities_to_docs(G)
    print(f"Converted {len(documents)} entities to documents.")

    # 3. Get or create knowledge base
    kb = get_kb(kb_name=kb_name_entity)

    # 4. Clear existing collection (optional)
    print(f"Clearing existing collection: {kb_name_entity}")
    kb.delete_collection()

    # 5. Add documents to KB
    print(f"Adding {len(documents)} documents to KB...")
    ids = kb.add_documents(documents)
    print(f"Successfully added {len(ids)} documents to KB.")
    return


def ingest_to_kb(
        graph_pkl_path: str,
        summary_pkl_path: str,
        kb_name_summary: str,
        kb_name_entity: str,
    ):
    ingest_summaries_to_kb(
        summary_pkl_path=summary_pkl_path,
        kb_name_summary=kb_name_summary,
    )
    ingest_entities_to_kb(
        graph_pkl_path=graph_pkl_path,
        kb_name_entity=kb_name_entity,
    )
    return

if __name__ == "__main__":
    # Example usage
    output_dir = "D:\\MyProjects\\ragIM\\data\\outputs"
    graph_pkl_path = os.path.join(output_dir, "raw_graph.pkl")
    summary_pkl_path = os.path.join(output_dir, "summary.pkl")

    # Ingest summaries into KB
    ingest_to_kb(
        graph_pkl_path=graph_pkl_path,
        summary_pkl_path=summary_pkl_path,
        kb_name_summary="kb_ibm_graph_summaries",
        kb_name_entity="kb_ibm_graph_entities",
    )