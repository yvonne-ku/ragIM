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

def run_semantic_split(
    input_path: str,
    buffer_size: int,
    threshold_percentile: int
) -> str:
    """
    运行语义分割
    """
    logger.info(f"开始语义分割: 输入路径={input_path}, 缓冲区大小={buffer_size}, 阈值百分比={threshold_percentile}")
    output_dir = str(os.path.join(settings.basic_settings.CHUNKS_PATH))
    from split_by_semantic import semantic_split
    try:
        semantic_split(
            input_path=input_path,
            output_dir=output_dir,
            buffer_size=buffer_size,
            threshold_percentile=threshold_percentile
        )
        output_path = str(os.path.join(output_dir, f"baseline_split_b:{buffer_size}_p:{threshold_percentile}.json"))
        logger.info(f"语义分割完成，输出路径: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"语义分割失败: {e}")
        raise


def run_ingest_to_kb(
    chunks_path: str,
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
            json_path=chunks_path,
            kb_name=kb_name,
            refresh_kb=refresh_kb
        )
        logger.info(f"知识库摄入完成，知识库名称: {kb_name}")
        return kb_name
    except Exception as e:
        logger.error(f"知识库摄入失败: {e}")
        raise


def run_retrieve_and_evaluate(
    chunks_path: str,
    kb_name: str,
    top_k: int,
) -> tuple[dict, float, float, float, float]:
    """
    运行检索和评估
    """
    logger.info(f"开始检索和评估: 知识库名称={kb_name}, top_k={top_k}")
    from retrieve_and_evaluate import run_evaluation
    try:
        query_results, avg_hit_rate, avg_mrr, avg_precision, avg_recall = run_evaluation(
            json_path=chunks_path,
            kb_name=kb_name,
            top_k=top_k,
        )
        logger.info(f"检索和评估完成，知识库名称: {kb_name}")
        return query_results, avg_hit_rate, avg_mrr, avg_precision, avg_recall
    except Exception as e:
        logger.error(f"检索和评估失败: {e}")
        raise


def main():
    """
    主函数，串联整个 RAG 流程，超参：
    - buffer_size: 1,2,3
    - threshold_percentile: 75,80,85,90,95
    - top_k: 3,5,7
    """
    input_path = str(os.path.join(settings.basic_settings.RAW_JSON_PATH, "ibm_all.json"))
    kb_name = "kb_baseline_ibm"

    try:
        for buffer_size in [1, 2, 3]:
            for threshold_percentile in [75, 80, 85, 90, 95]:
                for top_k in [3, 5, 7]:
                    chunks_path = run_semantic_split(
                        input_path=input_path,
                        buffer_size=buffer_size,
                        threshold_percentile=threshold_percentile
                    )
                    run_ingest_to_kb(
                        chunks_path=chunks_path,
                        kb_name=kb_name,
                        refresh_kb=True
                    )
                    query_results, avg_hit_rate, avg_mrr, avg_precision, avg_recall = run_retrieve_and_evaluate(
                        chunks_path=chunks_path,
                        kb_name=kb_name,
                        top_k=top_k
                    )

                    # Save Result
                    output_dir = os.path.join(settings.basic_settings.RESULTS_PATH)
                    os.makedirs(output_dir, exist_ok=True)
                    evaluation_result = {
                        "timestamp": time.strftime("%m_%d_%H_%M"),
                        "method": "baseline",
                        "kb_name": kb_name,
                        "params": {
                            "buffer_size": buffer_size,
                            "threshold_percentile": threshold_percentile,
                            "top_k": top_k,
                        },
                        "metrics": {
                            "avg_hit_rate": avg_hit_rate,
                            "avg_mrr": avg_mrr,
                            "avg_precision": avg_precision,
                            "avg_recall": avg_recall,
                        },
                        "query_results": query_results,
                    }
                    output_file = os.path.join(output_dir, f"baseline_b:{buffer_size}_p:{threshold_percentile}_k:{top_k}.json")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
                    logger.info(f"超参组合: buffer_size={buffer_size}, threshold_percentile={threshold_percentile}, top_k={top_k} 执行完成，输出路径: {output_file}")

        logger.info("整个 RAG 流程执行完成！")
    except Exception as e:
        logger.error(f"流程执行失败: {e}")
        raise

if __name__ == "__main__":
    main()