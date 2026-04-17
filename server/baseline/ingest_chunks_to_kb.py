import json
import os
import sys
from typing import List
from langchain_core.documents import Document

# Add project root to sys.path to import settings and services
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # ragIM
if project_root not in sys.path:
    sys.path.append(project_root)

from server import settings
from server.kb_service.chromadb_service import get_kb

def process_json_to_docs(json_path: str) -> List[Document]:
    """
    Load JSON chunks and convert them to LangChain Document objects.
    Each chunk (list of messages) becomes one Document.
    """
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    method = data.get("method", "unknown")
    window_id = data.get("window_id", "1")
    chunks = data.get("chunks", [])
    
    # Global docs List
    docs = []

    for i, chunk in enumerate(chunks):
        # 1. Concatenate all message texts in the chunk
        full_text = "\n".join([msg.get("text", "") for msg in chunk])
        # 2. Collect unique topic_ids in this chunk for metadata
        topic_ids = list(set([msg.get("topic_id", "") for msg in chunk]))        
        # 3. Create Document
        doc = Document(
            page_content=full_text,
            metadata={
                "source": json_path,
                "method": method,
                "window_id": window_id,
                "chunk_id": i,
                "topic_ids": topic_ids,
                "msg_count": len(chunk)
            }
        )
        docs.append(doc)
    
    print(f"Loaded {len(docs)} documents from {json_path} (Method: {method})")
    return docs

def ingest_to_kb(json_filename: str, kb_name: str, refresh_kb: bool = True):
    """
    Ingest documents into a specific knowledge base.
    """
    json_path = os.path.join(current_dir, json_filename)
    docs = process_json_to_docs(json_path)
    if not docs:
        return

    # 1. Get Singleton KB instance (Chromadb)
    kb = get_kb(kb_name=kb_name)
    
    # 2. Clear existing collection for a fresh start (Optional)
    if refresh_kb:
        try:
            kb.delete_collection()
            print(f"Deleted existing collection: {kb_name}")
            kb = get_kb(kb_name=kb_name) # Re-init
        except:
            print(f"Error deleting collection: {kb_name}")

    # 3. Add documents
    print(f"Ingesting documents into {kb_name}...")
    kb.add_documents(docs)
    print("Ingestion completed.")


if __name__ == "__main__":
    # Ingest Naive Baseline
    # For Only Ubuntu Dataset
    ingest_to_kb("ubuntu_naive_split.json", "kb_ubuntu_naive")
    
    # Ingest Semantic Baseline
    # For Only Ubuntu Dataset
    ingest_to_kb("ubuntu_semantic_split.json", "kb_ubuntu_semantic")
