import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Set, Tuple
import pandas as pd
import yaml
from graphrag.api.query import local_search
from graphrag.config.load_config import GraphRagConfig

from server import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_queries(queries_json_path: Path) -> List[Dict]:
    """加载查询文件，返回每个 query 的字典列表"""
    with open(queries_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["queries"]


def build_text_unit_to_chunk_map(output_dir: Path) -> dict:
    """
    建立 text_unit.id -> chunk_id 的映射
    """
    # 加载 documents 表和 text_units 表
    docs_df = pd.read_parquet(output_dir / "documents.parquet")
    tus_df = pd.read_parquet(output_dir / "text_units.parquet")

    # documents 表中 id 是原始文件名（如 "1.txt"），title 可能也是，这里假设 id 即为文件名
    # 建立 doc_id -> chunk_id 的映射（去掉 .txt 后缀）
    doc_to_chunk = {}
    for _, row in docs_df.iterrows():
        doc_id = str(row["id"])
        chunk_id = doc_id.replace(".txt", "")
        doc_to_chunk[doc_id] = chunk_id

    # 建立 text_unit.id -> chunk_id 的映射
    tu_to_chunk = {}
    for _, row in tus_df.iterrows():
        tu_id = row["id"]
        # document_ids 是一个列表（因为一个 text_unit 可能对应多个 doc），但你的自定义流程是一对一
        doc_ids = row.get("document_ids", [])
        if doc_ids:
            # 取第一个（也是唯一一个）document_id
            doc_id = doc_ids[0]
            chunk_id = doc_to_chunk.get(doc_id)
            if chunk_id:
                tu_to_chunk[tu_id] = chunk_id
    return tu_to_chunk


async def run_local_search_for_query(
    config: GraphRagConfig,
    entities_df: pd.DataFrame,
    communities_df: pd.DataFrame,
    community_reports_df: pd.DataFrame,
    text_units_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    query_text: str,
    top_k: int,
    tu_to_chunk: dict,
    covariates_df: pd.DataFrame = None,  # 新增一个可选参数
) -> List[str]:
    """
    对单个查询执行 local_search，返回 top_k 个 text_unit ID
    """
    # 创建一个空的 DataFrame，并包含一些必要但可为空的列，以确保兼容性
    if covariates_df is None:
        covariates_df = pd.DataFrame(
            columns=['id', 'subject_id', 'object_id', 'type', 'status', 'start_date', 'end_date', 'description',
                     'source_text', 'text_unit_ids', 'document_ids', 'attributes'])

    response, context_data = await local_search(
        config=config,
        entities=entities_df,
        communities=communities_df,
        community_reports=community_reports_df,
        text_units=text_units_df,
        relationships=relationships_df,
        query=query_text,
        community_level=2,           # 根据第几层的社区进行扩展检索
        response_type="Multiple Paragraphs",
        covariates=covariates_df,
    )

    # 从 context_data 中提取检索到的 text_unit
    # retrieved_units_df 已经按相关性降序排列，取前 top_k 的 'id' 列
    retrieved_units_df = context_data.get('text_units')
    if retrieved_units_df is None or retrieved_units_df.empty:
        logger.warning(f"No text units retrieved for query: {query_text[:50]}...")
        return []
    unit_ids = retrieved_units_df.head(top_k)['id'].tolist()

    # 从 text_unit 映射到 chunk_id
    chunk_ids = [tu_to_chunk.get(uid) for uid in unit_ids if uid in tu_to_chunk]
    return chunk_ids


async def evaluate_retrieval(
    queries: List[Dict],
    workspace_root: Path,
):
    """
    对所有查询进行评估，计算 Precision@K, Recall@K, F1@K
    """
    # 1. 加载配置和索引数据
    settings_path = workspace_root / "settings.yaml"
    with open(settings_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    config = GraphRagConfig.model_validate(config_dict)

    output_dir = workspace_root / "output"
    entities_df = pd.read_parquet(output_dir / "entities.parquet")
    communities_df = pd.read_parquet(output_dir / "communities.parquet")
    community_reports_df = pd.read_parquet(output_dir / "community_reports.parquet")
    text_units_df = pd.read_parquet(output_dir / "text_units.parquet")
    relationships_df = pd.read_parquet(output_dir / "relationships.parquet")

    # 2. 建立映射表
    tu_to_chunk = build_text_unit_to_chunk_map(output_dir)

    # 3. 为每个查询拿到 chunk_id 列表，之后为每个 K 截断计算指标
    top_k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    max_k = max(top_k_list)
    all_retrieved_chunks = []  # 与 queries 顺序一一对应
    for q in queries:
        query_text = q.get("query_text")
        retrieved = await run_local_search_for_query(
            config=config,
            entities_df=entities_df,
            communities_df=communities_df,
            community_reports_df=community_reports_df,
            text_units_df=text_units_df,
            relationships_df=relationships_df,
            query_text=query_text,
            top_k=max_k,
            tu_to_chunk=tu_to_chunk,
        )
        all_retrieved_chunks.append(retrieved)
        logger.info(f"Query {q['query_id']}: retrieved {len(retrieved)} chunks")

    # 3. 对每个 K 进行指标的截断计算
    for k in top_k_list:
        total_recall = 0
        total_precision = 0
        query_results = []

        for idx, q in enumerate(queries):
            query_id = q["query_id"]
            query_text = q["query_text"]
            ground_truth = set(q.get("answer_chunk_ids", []))
            if not ground_truth:
                logger.warning(f"Query {query_id} has no ground truth chunks, skipping")
                continue

            retrieved_ids = all_retrieved_chunks[idx][:k]  # 取前 k 个 chunk_id
            retrieved_set = set(retrieved_ids)

            tp = len(retrieved_set & ground_truth)
            precision = tp / k if k > 0 else 0.0
            recall = tp / len(ground_truth) if ground_truth else 0.0

            total_precision += precision
            total_recall += recall

            query_results.append({
                "query_id": query_id,
                "query": query_text,
                "answer_chunk_ids": list(ground_truth),
                "retrieved_ids": retrieved_ids,
                "precision": precision,
                "recall": recall
            })

        # 计算 k 取值下的总指标文件
        avg_precision = total_precision / len(queries)
        avg_recall = total_recall / len(queries)
        F1_score = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if avg_precision and avg_recall else 0

        output_dir = settings.basic_settings.RESULTS_DIR / f"graphrag_b_1_p_90_K_{k}.json"
        evaluation_result = {
            "timestamp": time.strftime("%m_%d_%H_%M"),
            "model": "graphrag",
            "kb_name": "lancedb",
            "params": {
                "buffer_size": 1,
                "threshold_percentile": 90,
                "top_k": k,
            },
            "metrics": {
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "F1_score": F1_score,
            },
            "query_results": query_results,
        }
        with open(output_dir, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        logger.info(
            f"超参组合: buffer_size={1}, threshold_percentile={90}, top_k={k} 执行完成，输出路径: {output_dir}")


if __name__ == "__main__":
    graphrag_base_dir = settings.basic_settings.SERVER_ROOT / "graph_rag_model"
    workspace_root = graphrag_base_dir / "workspace"
    queries_json_path = settings.basic_settings.CHUNKS_DIR / "queries_with_chunk_answer_semantic_b_1_p_90.json"

    # 加载查询
    queries = load_queries(queries_json_path)
    logger.info(f"Loaded {len(queries)} queries")

    # 运行评估
    asyncio.run(evaluate_retrieval(queries, workspace_root))
