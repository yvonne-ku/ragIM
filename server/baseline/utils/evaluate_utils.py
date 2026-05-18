import json
import os
from typing import List

from server import settings


def calculate_sources_metrics(retrieved_ids: List[str], ground_truth_id: List[str]):
    correct_retrievals = len(set(retrieved_ids) & set(ground_truth_id))
    precision = correct_retrievals / len(retrieved_ids) if retrieved_ids else 0
    recall = correct_retrievals / len(ground_truth_id) if ground_truth_id else 0
    return precision, recall


def evaluate_retrieval(retriever):
    # 1. Evaluation Set
    with open(os.path.join(settings.basic_settings.CHUNKS_DIR, "queries_with_chunk_answer_semantic_b_1_p_90.json"), 'r', encoding='utf-8') as f:
        query_list = json.load(f)['queries']


    # Evaluation Metrics:
    # hit_rate (for one source query): Whether the target chunk is in the top-k results
    # mRR (for one source query): The reciprocal rank of the first correct prediction
    # precision: The proportion of correct predictions among all predictions
    # recall: The proportion of correct predictions that are among all the predictions
    avg_precision = 0
    avg_recall = 0
    F1_score = 0
    query_results = []

    # 2. Retrieve
    total_recall = 0
    total_precision = 0
    print(f"\n--- Evaluating Retrieval: ---")
    for item in query_list:
        query_text = item['query_text']
        target = [str(cid) for cid in item['answer_chunk_ids']]

        # Retrieve results
        results = retriever.retrieve(query_text)
        retrieved_ids = [str(res['chunk_id']) for res in results]

        # Evaluate
        precision, recall = calculate_sources_metrics(retrieved_ids, target)
        total_recall += recall
        total_precision += precision

        # Query_Results
        query_results.append({
            "query_id": item['query_id'],
            "query": query_text,
            "answer_chunk_ids": target,
            "retrieved_ids": retrieved_ids,
            "precision": precision,
            "recall": recall
        })

    # 3. Calculate average metrics
    avg_precision = total_precision / len(query_list)
    avg_recall = total_recall / len(query_list)
    F1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if avg_precision and avg_recall else 0

    return query_results, avg_precision, avg_recall, F1_score

