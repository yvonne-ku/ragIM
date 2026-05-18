import json
import logging
import os
import time
from server import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_ingest_to_kb(
    CHUNKS_DIR: str,
    kb_name: str,
    refresh_kb: bool
) -> str:
    """
    运行知识库摄入
    """
    logger.info(f"开始知识库摄入: 知识库名称={kb_name}, 刷新知识库={refresh_kb}")
    from ingest_chunks_to_kb import ingest_to_kb
    try:
        ingest_to_kb(
            json_path=CHUNKS_DIR,
            kb_name=kb_name,
            refresh_kb=refresh_kb
        )
        logger.info(f"知识库摄入完成，知识库名称: {kb_name}")
        return kb_name
    except Exception as e:
        logger.error(f"知识库摄入失败: {e}")
        raise


def run_retrieve_and_evaluate(
    CHUNKS_DIR: str,
    kb_name: str,
    top_k: int,
) -> tuple[dict, float, float, float]:
    """
    运行检索和评估
    """
    logger.info(f"开始检索和评估: 知识库名称={kb_name}, top_k={top_k}")
    from retrieve_and_evaluate import run_evaluation
    try:
        query_results, avg_precision, avg_recall, F1_score = run_evaluation(
            json_path=CHUNKS_DIR,
            kb_name=kb_name,
            top_k=top_k,
        )
        logger.info(f"检索和评估完成，知识库名称: {kb_name}")
        return query_results, avg_precision, avg_recall, F1_score
    except Exception as e:
        logger.error(f"检索和评估失败: {e}")
        raise


def main():
    """
    - buffer_size: 1,2,3
    - threshold_percentile: 75,80,85,90,95
    - top_k: 3,5,7
    """
    CHUNKS_DIR = str(os.path.join(settings.basic_settings.CHUNKS_DIR, "semantic_split_b_1_p_90.json"))
    kb_name = "kb_baseline"

    try:
        run_ingest_to_kb(
            CHUNKS_DIR=CHUNKS_DIR,
            kb_name=kb_name,
            refresh_kb=False        # 不需要重新生成向量库就设置为 False
        )

        for top_k in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
            query_results, avg_precision, avg_recall, F1_score = run_retrieve_and_evaluate(
                CHUNKS_DIR=CHUNKS_DIR,
                kb_name=kb_name,
                top_k=top_k
            )

            # Save Result
            output_dir = str(os.path.join(settings.basic_settings.RESULTS_DIR))
            os.makedirs(output_dir, exist_ok=True)
            output_file = str(os.path.join(output_dir, f"baseline_b_1_p_90_K_{top_k}.json"))
            evaluation_result = {
                "timestamp": time.strftime("%m_%d_%H_%M"),
                "model": "baseline",
                "kb_name": kb_name,
                "params": {
                    "buffer_size": 1,
                    "threshold_percentile": 90,
                    "top_k": top_k,
                },
                "metrics": {
                    "avg_precision": avg_precision,
                    "avg_recall": avg_recall,
                    "F1_score": F1_score,
                },
                "query_results": query_results,
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
            logger.info(f"超参组合: buffer_size={1}, threshold_percentile={90}, top_k={top_k} 执行完成，输出路径: {output_file}")

        logger.info("整个 RAG 流程执行完成！")
    except Exception as e:
        logger.error(f"流程执行失败: {e}")
        raise

if __name__ == "__main__":
    main()