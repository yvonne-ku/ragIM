import os
import json
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from server import settings
from server.kb_singleton_util import get_kb
from server.evaluate_utils import evaluate_retrieval


class HybridRetriever:
    """
    Hybrid Retriever: Vector + BM25
    Uses standard Reciprocal Rank Fusion (RRF) for combining results.
    """

    def __init__(self, kb_name: str, documents: List[dict], top_k: int):
        self.kb = get_kb(kb_name=kb_name)
        self.raw_docs = documents
        self.top_k = top_k

        # Initialize BM25
        # Tokenize and normalize the text into a list of tokens(split by blank space)
        tokenized_corpus = [doc['text'].lower().split() for doc in self.raw_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Combined search using Vector similarity and BM25.
        Uses standard Reciprocal Rank Fusion (RRF) algorithm.
        """
        k_rrf = 60

        # 1. Vector Search
        vector_results = self.kb.search(query, top_k=self.top_k)

        # 2. BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # 防止文档总数少于 k_val 导致越界
        actual_k = min(self.top_k, len(bm25_scores))
        bm25_ranked_indices = bm25_scores.argsort()[::-1][:actual_k]
        bm25_results = [self.raw_docs[i] for i in bm25_ranked_indices]

        # 3. Combine results using RRF
        combined_results = {}

        # 3.1 Add Vector results
        for i, res in enumerate(vector_results):
            chunk_id = res.get('chunk_id', f'chunk_{i}')
            rank = i + 1
            combined_results[chunk_id] = {
                'text': res.get('text', ''),
                'chunk_id': chunk_id,
                'score': 1.0 / (k_rrf + rank)
            }

        # 3.2 Add BM25 results
        for i, res in enumerate(bm25_results):
            chunk_id = res['metadata']['chunk_id']
            rank = i + 1
            rrf_score = 1.0 / (k_rrf + rank)

            if chunk_id not in combined_results:
                combined_results[chunk_id] = {
                    'text': res['text'],
                    'chunk_id': chunk_id,
                    'score': rrf_score
                }
            else:
                combined_results[chunk_id]['score'] += rrf_score

        # 3.3 Sort by combined RRF score
        sorted_results = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)[:self.top_k]
        return sorted_results


def run_evaluation(json_path: str, kb_name: str, top_k: int = 5):
    if not os.path.exists(json_path):
        return
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Prepare documents for BM25
    documents = []
    for chunk in data['chunks']:
        text = "\n".join([msg['text'] for msg in chunk['messages']])
        documents.append({"text": text, "metadata": {"chunk_id": chunk['chunk_id'], "method": data['method']}})

    # 2. Initialize retriever
    retriever = HybridRetriever(kb_name, documents, top_k)

    # 3. Evaluate retrieval
    return evaluate_retrieval(retriever)


if __name__ == "__main__":
    # Ensure ingest_to_kb.py has been run first to populate ChromaDB
    run_evaluation(os.path.join(settings.basic_settings.CHUNKS_PATH, "ubuntu_naive_split.json"), "kb_ubuntu_naive")
    run_evaluation(os.path.join(settings.basic_settings.CHUNKS_PATH, "ubuntu_semantic_split.json"), "kb_ubuntu_semantic")