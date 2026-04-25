import os
import json
from typing import List, Dict
from rank_bm25 import BM25Okapi
from server import settings
from server.kb_singleton_util import get_kb


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

    def retrieve(self, query: str):
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


def calculate_single_source_metrics(retrieved_ids: List[str], ground_truth_id: str):
    hit = 1 if ground_truth_id in retrieved_ids else 0
    mrr = 0
    if hit:
        rank = retrieved_ids.index(ground_truth_id) + 1
        mrr = 1.0 / rank
    return hit, mrr

def calculate_multiple_sources_metrics(retrieved_ids: List[str], ground_truth_id: List[str]):
    correct_retrievals = len(set(retrieved_ids) & set(ground_truth_id))
    precision = correct_retrievals / len(retrieved_ids) if retrieved_ids else 0
    recall = correct_retrievals / len(ground_truth_id) if ground_truth_id else 0
    return precision, recall


def run_evaluation(json_path: str, kb_name: str, top_k: int = 5, alpha: float = 0.5):
    if not os.path.exists(json_path):
        return [], 0.0, 0.0, 0.0, 0.0
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Prepare documents for BM25
    documents = []
    for chunk in data['chunks']:
        text = "\n".join([msg['text'] for msg in chunk['messages']])
        documents.append({"text": text, "metadata": {"chunk_id": chunk['chunk_id'], "method": data['method']}})

    # 2. Evaluation Set
    single_eval_set = [
        {"type":"single", "query_id":"query_001", "query": "java 1.4 is slow", "target_chunk_id": ["chunk_00001"]},
        {"type":"single", "query_id":"query_002", "query": "how to install jre", "target_chunk_id": ["chunk_00001"]},
    ]
    multi_eval_set = [
        {"type":"multi", "query_id":"query_003", "query": "java 1.4 is slow", "target_chunk_id": ["chunk_00001", "chunk_00002"]},
    ]

    # Evaluation Metrics:
    # hit_rate (for one source query): Whether the target chunk is in the top-k results
    # mRR (for one source query): The reciprocal rank of the first correct prediction
    # precision: The proportion of correct predictions among all predictions
    # recall: The proportion of correct predictions that are among all the predictions
    avg_hit_rate = 0
    avg_mrr = 0
    avg_precision = 0
    avg_recall = 0
    query_results = []
    retriever = HybridRetriever(kb_name, documents, top_k, alpha)

    # 3. Evaluate Retrieval

    # 3.1 Single Eval Test
    total_hit_rate = 0
    total_mrr = 0
    print(f"\n--- Evaluating Retrieval for {kb_name} for Single Source Queries: ---")
    for item in single_eval_set:
        query = item['query']
        target = item['target_chunk_id']

        # Retrieve results
        results = retriever.retrieve(query)
        retrieved_ids = [res['chunk_id'] for res in results]

        # Evaluate
        hit, mrr = calculate_single_source_metrics(retrieved_ids, target[0])
        total_hit_rate += hit
        total_mrr += mrr

        # Query_Results
        query_results.append({
            "query_id": item['query_id'],
            "query": query,
            "target_chunk_id": target,
            "retrieved_ids": retrieved_ids,
            "hit": hit,
            "mrr": mrr
        })

        # Calculate average metrics
        avg_hit_rate = total_hit_rate / len(single_eval_set)
        avg_mrr = total_mrr / len(single_eval_set)

    # 3.2 Multi Eval Test
    total_recall = 0
    total_precision = 0
    print(f"\n--- Evaluating Retrieval for {kb_name} for Multiple Sources Queries: ---")
    for item in multi_eval_set:
        query = item['query']
        target = item['target_chunk_id']

        # Retrieve results
        results = retriever.retrieve(query)
        retrieved_ids = [res['chunk_id'] for res in results]

        # Evaluate
        precision, recall = calculate_multiple_sources_metrics(retrieved_ids, target)
        total_recall += recall
        total_precision += precision

        # Query_Results
        query_results.append({
            "query_id": item['query_id'],
            "query": query,
            "target_chunk_id": target,
            "retrieved_ids": retrieved_ids,
            "precision": precision,
            "recall": recall
        })

        # Calculate average metrics
        avg_precision = total_precision / len(multi_eval_set)
        avg_recall = total_recall / len(multi_eval_set)

    return query_results, avg_hit_rate, avg_mrr, avg_precision, avg_recall

if __name__ == "__main__":
    # Ensure ingest_to_kb.py has been run first to populate ChromaDB
    run_evaluation(os.path.join(settings.basic_settings.CHUNKS_PATH, "ubuntu_naive_split.json"), "kb_ubuntu_naive")
    run_evaluation(os.path.join(settings.basic_settings.CHUNKS_PATH, "ubuntu_semantic_split.json"), "kb_ubuntu_semantic")