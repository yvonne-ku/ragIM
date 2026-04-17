import json
import os
import sys
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # ragIM
if project_root not in sys.path:
    sys.path.append(project_root)

from server.kb_service.chromadb_service import get_kb

class HybridRetriever:
    """
    Demonstrates Hybrid Retrieval (Vector + BM25)
    """
    def __init__(self, kb_name: str, documents: List[dict]):
        self.kb = get_kb(kb_name=kb_name)
        self.raw_docs = documents # List of {text: ..., metadata: ...}
        
        # Initialize BM25
        tokenized_corpus = [doc['text'].lower().split() for doc in self.raw_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5, vector_weight: float = 0.5):
        """
        Combined search using Vector similarity and BM25.
        Uses simple weighted sum or Reciprocal Rank Fusion (RRF).
        """
        # 1. Vector Search
        vector_results = self.kb.search(query, top_k=top_k*2)
        
        # 2. BM25 Keyword Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k*2]
        
        # 3. Reciprocal Rank Fusion (RRF) - A robust way to combine rankings
        rrf_scores = {} # doc_id or content_hash -> score
        
        # Add Vector results to RRF
        for rank, doc in enumerate(vector_results):
            doc_id = doc.metadata.get('chunk_id')
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)
            
        # Add BM25 results to RRF
        for rank, idx in enumerate(bm25_top_indices):
            doc_id = idx # chunk_id in our case
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (60 + rank)
            
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        final_results = []
        for doc_id, score in sorted_ids:
            final_results.append({
                "chunk_id": doc_id,
                "text": self.raw_docs[doc_id]['text'],
                "metadata": self.raw_docs[doc_id]['metadata'],
                "rrf_score": score
            })
            
        return final_results

def calculate_metrics(retrieved_ids: List[int], ground_truth_id: int):
    """
    Calculate Recall@K and MRR for a single query.
    """
    hit = 1 if ground_truth_id in retrieved_ids else 0
    mrr = 0
    if hit:
        rank = retrieved_ids.index(ground_truth_id) + 1
        mrr = 1.0 / rank
    return hit, mrr

def run_evaluation_demo(json_filename: str, kb_name: str):
    json_path = os.path.join(current_dir, json_filename)
    if not os.path.exists(json_path):
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Prepare documents for BM25
    documents = []
    for i, chunk in enumerate(data['chunks']):
        text = "\n".join([msg['text'] for msg in chunk])
        documents.append({"text": text, "metadata": {"chunk_id": i, "method": data['method']}})

    retriever = HybridRetriever(kb_name, documents)
    
    # Simulation: Let's assume some queries and their target chunk_id
    # In a real scenario, you'd have a curated evaluation set.
    eval_set = [
        {"query": "java 1.4 is slow", "target_chunk_id": 0},
        {"query": "how to install jre", "target_chunk_id": 0},
    ]
    
    total_recall = 0
    total_mrr = 0
    
    print(f"\n--- Evaluating Retrieval for {kb_name} ---")
    for item in eval_set:
        query = item['query']
        target = item['target_chunk_id']
        
        results = retriever.retrieve(query, top_k=5)
        retrieved_ids = [res['chunk_id'] for res in results]
        
        hit, mrr = calculate_metrics(retrieved_ids, target)
        total_recall += hit
        total_mrr += mrr
        
        print(f"Query: {query} | Target Chunk: {target} | Hit: {hit} | MRR: {mrr:.4f}")

    avg_recall = total_recall / len(eval_set)
    avg_mrr = total_mrr / len(eval_set)
    
    print(f"\nFinal Metrics for {data['method']}:")
    print(f"Recall@5: {avg_recall:.4f}")
    print(f"MRR: {avg_mrr:.4f}")

if __name__ == "__main__":
    # Ensure ingest_to_kb.py has been run first to populate ChromaDB
    run_evaluation_demo("ubuntu_naive_split.json", "kb_ubuntu_naive")
    run_evaluation_demo("ubuntu_semantic_split.json", "kb_ubuntu_semantic")
