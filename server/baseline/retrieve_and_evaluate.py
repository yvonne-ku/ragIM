import os
import json
import time
from typing import List, Dict
from rank_bm25 import BM25Okapi
from server import settings
from server.kb_singleton_util import get_kb

class HybridRetriever:

    """
    Hybrid Retriever: Vector + BM25
    """
    def __init__(self, kb_name: str, documents: List[dict]):
        self.kb = get_kb(kb_name=kb_name)
        self.raw_docs = documents

        # Initialize BM25
        # Tokenize and normalize the text into a list of tokens(split by blank space)
        tokenized_corpus = [doc['text'].lower().split() for doc in self.raw_docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5):
        """
        Combined search using Vector similarity and BM25.
        Uses simple weighted sum or Reciprocal Rank Fusion (RRF).
        """
        # Vector Search
        vector_results = self.kb.search(query, top_k=top_k)

        # BM25 Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked_indices = bm25_scores.argsort()[::-1][:top_k]
        bm25_results = [self.raw_docs[i] for i in bm25_ranked_indices]

        # Combine results (simple merging for now)
        combined_results = {}

        # Add vector results
        for i, res in enumerate(vector_results):
            chunk_id = res.get('chunk_id', f'chunk_{i}')
            combined_results[chunk_id] = {
                'text': res.get('text', ''),
                'chunk_id': chunk_id,
                'score': 1.0 / (i + 1)  # Higher rank = higher score
            }

        # Add BM25 results
        for i, res in enumerate(bm25_results):
            chunk_id = res['metadata']['chunk_id']
            if chunk_id not in combined_results:
                combined_results[chunk_id] = {
                    'text': res['text'],
                    'chunk_id': chunk_id,
                    'score': 0.5 / (i + 1)  # Lower weight for BM25
                }
            else:
                # Boost score if chunk appears in both
                combined_results[chunk_id]['score'] += 0.5 / (i + 1)

        # Sort by score
        sorted_results = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)[:top_k]
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

"""
- top_k
"""
def run_evaluation_for_baselines(json_path: str, kb_name: str):
    if not os.path.exists(json_path):
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare documents for BM25
    documents = []
    for chunk in data['chunks']:
        text = "\n".join([msg['text'] for msg in chunk['messages']])
        documents.append({"text": text, "metadata": {"chunk_id": chunk['chunk_id'], "method": data['method']}})

    # Initialize retriever
    retriever = HybridRetriever(kb_name, documents)


    # TODO test dataset
    eval_set = [
        {"type":"single", "query_id":"query_001", "query": "java 1.4 is slow", "target_chunk_id": ["chunk_00001"]},
        {"type":"single", "query_id":"query_002", "query": "how to install jre", "target_chunk_id": ["chunk_00001"]},
    ]

    # Evaluation Metrics:
    # hit_rate (for one source query): Whether the target chunk is in the top-k results
    # mRR (for one source query): The reciprocal rank of the first correct prediction
    # precision: The proportion of correct predictions among all predictions
    # recall: The proportion of correct predictions that are among all the predictions

    top_k = 5
    
    # 初始化评估指标变量
    avg_hit_rate = 0
    avg_mrr = 0
    avg_precision = 0
    avg_recall = 0
    query_results = []

    # In the end, there will be kb_naive, kb_semantic, kb_clustering and kb_graph. Every kb has been ingested by the whole dataset.
    # And there will be two kinds of query tests, one has queries from the single source, and the other has queries from multiple sources.
    # there two kinds should be evaluated separately.
    if (eval_set[0]['type'] == 'single'):
        total_hit_rate = 0
        total_mrr = 0
        query_results = []
        print(f"\n--- Evaluating Retrieval for {kb_name} for Single Source Queries: ---")
        for item in eval_set:
            query = item['query']
            target = item['target_chunk_id']

            # Retrieve results
            results = retriever.retrieve(query, top_k=top_k)
            retrieved_ids = [res['chunk_id'] for res in results]

            # Evaluate
            hit, mrr = calculate_single_source_metrics(retrieved_ids, target[0])
            total_hit_rate += hit
            total_mrr += mrr
            print(f"Query: {query} | Target Chunk: {target} | Hit: {hit} | MRR: {mrr:.4f}")
            
            # Query_Results
            query_results.append({
                "query": query,
                "target_chunk_id": target,
                "retrieved_ids": retrieved_ids,
                "hit": hit,
                "mrr": mrr
            })

        # Calculate average metrics
        avg_hit_rate = total_hit_rate / len(eval_set)
        avg_mrr = total_mrr / len(eval_set)
        print(f"\nFinal Metrics for {data['method']}:")
        print(f"Hit Rate@5: {avg_hit_rate:.4f}")
        print(f"MRR: {avg_mrr:.4f}")

    elif (eval_set[0]['type'] == 'multi'):
        total_recall = 0
        total_precision = 0
        query_results = []
        print(f"\n--- Evaluating Retrieval for {kb_name} for Multiple Sources Queries: ---")
        for item in eval_set:
            query = item['query']
            target = item['target_chunk_id']

            # Retrieve results
            results = retriever.retrieve(query, top_k=top_k)
            retrieved_ids = [res['chunk_id'] for res in results]

            # Evaluate
            precision, recall = calculate_multiple_sources_metrics(retrieved_ids, target)
            total_recall += recall
            total_precision += precision
            print(f"Query: {query} | Target Chunk: {target} | Precision: {precision} | Recall: {recall:.4f}")
            
            # Query_Results
            query_results.append({
                "query": query,
                "target_chunk_id": target,
                "retrieved_ids": retrieved_ids,
                "precision": precision,
                "recall": recall
            })

        # Calculate average metrics
        avg_precision = total_precision / len(eval_set)
        avg_recall = total_recall / len(eval_set)
        print(f"\nFinal Metrics for {data['method']}:")
        print(f"Precision: {avg_precision:.4f}")
        print(f"Recall: {avg_recall:.4f}")
    
    # Save Result
    output_dir = os.path.join(settings.basic_settings.OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%m_%d_%H_%M")
    method_name = data['method'].split('_')[0]
    evaluation_result = {
        "timestamp": timestamp,
        "kb_name": kb_name,
        "method": data['method'],
        "top_k": top_k,
        "eval_set_size": len(eval_set),
        "eval_type": eval_set[0]['type'],
        "metrics": {
            "hit_rate": avg_hit_rate if eval_set[0]['type'] == 'single' else None,
            "mrr": avg_mrr if eval_set[0]['type'] == 'single' else None,
            "precision": avg_precision if eval_set[0]['type'] == 'multi' else None,
            "recall": avg_recall if eval_set[0]['type'] == 'multi' else None
        },
        "query_results": query_results,
        "metadata": {
            "retriever": "HybridRetriever",
            "vector_store": "ChromaDB",
            "bm25_implementation": "rank_bm25"
        }
    }
    
    output_file = os.path.join(output_dir, f"{eval_set[0]['type']}_{method_name}_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation results saved to: {output_file}")

if __name__ == "__main__":
    # Ensure ingest_to_kb.py has been run first to populate ChromaDB
    run_evaluation_for_baselines(os.path.join(settings.basic_settings.CHUNKS_PATH, "ubuntu_naive_split.json"), "kb_ubuntu_naive")
    run_evaluation_for_baselines(os.path.join(settings.basic_settings.CHUNKS_PATH, "ubuntu_semantic_split.json"), "kb_ubuntu_semantic")