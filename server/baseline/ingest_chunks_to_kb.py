import json
import os
from typing import List
from langchain_core.documents import Document

from server import settings
from server.kb_singleton_util import get_kb

def process_chunks_to_docs(json_path: str) -> List[Document]:
    """
    Load JSON chunks and convert them to LangChain Document objects.
    Each chunk (list of messages) becomes one Document.
    """
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    window_id = data.get("window_id", "1")
    method = data.get("method", "unknown")
    chunks = data.get("chunks", [])
    
    # Global docs List
    docs = []
    for i, chunk in enumerate(chunks):
        full_text = chunk["concat_text"]
        doc = Document(
            page_content=full_text,
            metadata={
                "chunk_id": chunk["chunk_id"],
            }
        )
        docs.append(doc)
    print(f"Loaded {len(docs)} documents from {json_path} (Method: {method})")
    return docs

def ingest_to_kb(json_path: str, kb_name: str, refresh_kb: bool = True, batch_size: int = 8):
    """
    Ingest documents into a specific knowledge base.
    """
    docs = process_chunks_to_docs(json_path)
    if not docs:
        return

    if refresh_kb:
        # 1. Get Singleton KB instance (Chromadb)
        kb = get_kb(kb_name=kb_name)

        # 2. Clear existing collection for a fresh start (Optional)
        kb.delete_collection()
        print(f"Deleted existing collection: {kb_name}")

        # 3. Add documents in batches to avoid memory issues
        print(f"Ingesting documents into {kb_name}...")
        total_docs = len(docs)
        for i in range(0, total_docs, batch_size):
            batch = docs[i:i+batch_size]
            kb.add_documents(batch)
            print(f"Ingested {min(i+batch_size, total_docs)}/{total_docs} documents...")

    else:
        kb = get_kb(kb_name=kb_name)
        print(f"Use existing collection: {kb_name} Without Refresh Or Ingest")

    print("Ingestion completed.")


if __name__ == "__main__":
    # Ingest Naive Baseline
    # For Only Ubuntu Dataset
    ingest_to_kb(os.path.join(settings.basic_settings.CHUNKS_DIR, "ubuntu_naive_split.json"), "kb_ubuntu_naive")
    
    # Ingest Semantic Baseline
    # For Only Ubuntu Dataset
    ingest_to_kb(os.path.join(settings.basic_settings.CHUNKS_DIR, "ubuntu_semantic_split.json"), "kb_ubuntu_semantic")