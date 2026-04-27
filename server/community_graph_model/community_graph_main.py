"""
Main script to run the entire community graph pipeline
"""

import logging
import os
import json
from datetime import time

from streamlit import success

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
    output_path = str(os.path.join(output_dir, f"semantic_split_b:{buffer_size}_p:{threshold_percentile}.json"))
    if os.path.exists(output_path):
        logger.info(f"语义分割结果已存在，输出路径: {output_path}")
        return output_path

    from server.split_by_semantic import semantic_split
    try:
        output_path = semantic_split(
            input_path=input_path,
            output_dir=output_dir,
            buffer_size=buffer_size,
            threshold_percentile=threshold_percentile
        )
        logger.info(f"语义分割完成，输出路径: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"语义分割失败: {e}")
        raise

def run_build_graph_with_community(
        json_path: str,
        resolution_parameter: float,
        rebuild_graph: bool
) -> str:
    """
    运行社区图构建
    """
    output_dir = settings.basic_settings.OUTPUT_PATH
    logger.info(f"开始构建社区图: 输出目录={output_dir}, 分区参数={resolution_parameter}")
    from build_graph import build_graph
    try:
        pkl_path = build_graph(
            json_path=json_path,
            output_dir=output_dir,
            resolution_parameter=resolution_parameter,
            rebuild_graph=rebuild_graph
        )
        logger.info(f"社区图构建完成，输出路径: {pkl_path}")
        return pkl_path
    except Exception as e:
        logger.error(f"构建社区图失败：{e}")
        raise


def run_generate_community_summaries(
        graph_file: str,
        json_path: str,
) -> str:
    """
    运行社区摘要生成
    """
    output_dir = settings.basic_settings.OUTPUT_PATH
    logger.info(f"开始生成社区摘要：输出目录={output_dir}")
    from generate_community_summay import generate_summaries
    try:
        pkl_path = generate_summaries(
            graph_pkl_path=graph_file,
            json_path=json_path,
            output_dir=output_dir
        )
        logger.info(f"社区摘要生成完成，输出路径: {pkl_path}")
        return pkl_path
    except Exception as e:
        logger.error(f"生成社区失败：{e}")
        raise


def run_ingest_to_kb(
        graph_pkl_path: str,
        summary_pkl_path: str,
        kb_name_summary: str,
        kb_name_entity: str
) -> None:
    """
    运入社区摘要到知识库
    """
    logger.info(f"开始导入社区摘要到知识库: {kb_name_summary}")
    from ingest_community_graph_to_kb import ingest_to_kb
    try:
        ingest_to_kb(
            graph_pkl_path=graph_pkl_path,
            summary_pkl_path=summary_pkl_path,
            kb_name_summary=kb_name_summary,
            kb_name_entity=kb_name_entity
        )
        logger.info(f"社区摘要导入完成，摘要知识库名称: {kb_name_summary}, 实体知识库名称: {kb_name_entity}")
    except Exception as e:
        logger.error(f"导入摘要到知识库失败：{e}")
        raise


def run_graph_evaluation(
        kb_summary_name: str,
        kb_entity_name: str,
        graph_pkl_path: str,
        top_k_comm: int,
        top_k: int,
) -> tuple[dict, float, float, float, float]:
    """
    运行社区图评估
    """
    logger.info(f"开始评估社区图: {kb_summary_name}, {kb_entity_name}")
    from retrieve_and_evaluate import run_evaluation
    try:
        query_results, avg_hit_rate, avg_mrr, avg_precision, avg_recall = run_evaluation(
            kb_summary_name=kb_summary_name,
            kb_entity_name=kb_entity_name,
            graph_pkl_path=graph_pkl_path,
            top_k_comm=top_k_comm,
            top_k=top_k
        )
        logger.info(f"社区图评估完成，摘要知识库名称: {kb_summary_name}, 实体知识库名称: {kb_entity_name}")
        return query_results, avg_hit_rate, avg_mrr, avg_precision, avg_recall
    except Exception as e:
        logger.error(f"评估社区图失败：{e}")
        raise

def main():
    """
    - buffer_size: 1,2,3
    - threshold_percentile: 75,80,85,90,95
    - resolution_parameter: 0.8,1.0,1.2
    - top_k_comm: 1,2,3
    - top_k: 3,5,7

    优化：
    - 实体提取 Prompt
    """
    input_path = str(os.path.join(settings.basic_settings.RAW_JSON_PATH, "ibm_all.json"))
    kb_name_summary = "kb_summaries_ibm"
    kb_name_entity = "kb_entities_ibm"

    try:
        for buffer_size in [1, 2, 3]:
            for threshold_percentile in [75, 80, 85, 90, 95]:
                for resolution_parameter in [0.8, 1.0, 1.2]:
                    for top_k_comm in [1, 2, 3]:
                        for top_k in [3, 5, 7]:
                            chunk_path = run_semantic_split(
                                input_path=input_path,
                                buffer_size=buffer_size,
                                threshold_percentile=threshold_percentile
                            )
                            graph_pkl_path = run_build_graph_with_community(
                                json_path=chunk_path,
                                resolution_parameter=resolution_parameter,
                                rebuild_graph=True
                            )
                            summary_pkl_path = run_generate_community_summaries(
                                graph_file=graph_pkl_path,
                                json_path=chunk_path
                            )
                            run_ingest_to_kb(
                                graph_pkl_path=graph_pkl_path,
                                summary_pkl_path=summary_pkl_path,
                                kb_name_summary=kb_name_summary,
                                kb_name_entity=kb_name_entity
                            )
                            query_results, avg_hit_rate, avg_mrr, avg_precision, avg_recall = run_graph_evaluation(
                                kb_summary_name=kb_name_summary,
                                kb_entity_name=kb_name_entity,
                                graph_pkl_path=graph_pkl_path,
                                top_k_comm=top_k_comm,
                                top_k=top_k
                            )

                            # Save Result
                            output_dir = str(os.path.join(settings.basic_settings.RESULTS_PATH))
                            os.makedirs(output_dir, exist_ok=True)
                            output_file = str(os.path.join(output_dir, f"graph_b:{buffer_size}_p:{threshold_percentile}_r:{resolution_parameter}_ck:{top_k_comm}_k:{top_k}.json"))
                            evaluation_result = {
                                "timestamp": time.strftime("%m_%d_%H_%M"),
                                "method": "graph",
                                "kb_name": [kb_name_summary, kb_name_entity],
                                "params": {
                                    "buffer_size": buffer_size,
                                    "threshold_percentile": threshold_percentile,
                                    "resolution_parameter": resolution_parameter,
                                    "top_k_comm": top_k_comm,
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
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
                            logger.info(
                                f"超参组合: buffer_size={buffer_size}, threshold_percentile={threshold_percentile}, resolution_parameter={resolution_parameter}, top_k_comm={top_k_comm}, top_k={top_k} 执行完成，输出路径: {output_file}")

        logger.info("整个 RAG 流程执行完成！")
    except Exception as e:
        logger.error(f"流程执行失败: {e}")
        raise


if __name__ == "__main__":
    main()
