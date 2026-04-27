from typing import List


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


def evaluate_retrieval(retriever):
    # 1. Evaluation Set
    single_eval_set = [
        {"type": "single", "query_id": "query_001", "query": "java 1.4 is slow", "target_chunk_id": ["chunk_00001"]},
        {"type": "single", "query_id": "query_002", "query": "how to install jre", "target_chunk_id": ["chunk_00001"]},
    ]
    multi_eval_set = [
        {"type": "multi", "query_id": "query_003", "query": "java 1.4 is slow",
         "target_chunk_id": ["chunk_00001", "chunk_00002"]},
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


    # 2. Single Eval Test
    total_hit_rate = 0
    total_mrr = 0
    print(f"\n--- Evaluating Retrieval for Single Source Queries: ---")
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


    # 3. Multi Eval Test
    total_recall = 0
    total_precision = 0
    print(f"\n--- Evaluating Retrieval for Multiple Sources Queries: ---")
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

