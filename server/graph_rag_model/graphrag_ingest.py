import json
import logging
from pathlib import Path

import yaml
from server import settings
from server.graph_rag_model.patches.custom_create_base_text_units import run_workflow as custom_chunk_workflow
import os

logger = logging.getLogger(__name__)

def update_graphrag_settings(workspace_dir: Path):
    settings_path = workspace_dir / "settings.yaml"
    with open(settings_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 1. 分块策略
    config['chunks'] = {
        "size": 1000000,
        "overlap": 0,
    }

    # 2. 文本生成模型
    config['models']['default_chat_model'] = {
        "type": "chat",
        "model_provider": "openai",
        "model": "glm-4",
        "auth_type": "api_key",
        "api_key": settings.api_model_settings.MODEL_PLATFORMS["zhipuai"].api_key,
        "api_base": settings.api_model_settings.MODEL_PLATFORMS["zhipuai"].base_url,
        "concurrent_requests": 5,
        "async_mode": "threaded",
        "retry_strategy": "exponential_backoff",
        "max_retries": 5,
    }

    # 3. 嵌入模型
    config['models']['default_embedding_model'] = {
        "type": "embedding",
        "model_provider": "openai",
        "model": "embedding-2",
        "auth_type": "api_key",
        "api_key": settings.api_model_settings.MODEL_PLATFORMS["zhipuai"].api_key,
        "api_base": settings.api_model_settings.MODEL_PLATFORMS["zhipuai"].base_url,
        "concurrent_requests": 5,
        "async_mode": "threaded",
        "retry_strategy": "exponential_backoff",
        "max_retries": 5,
    }

    # 4. 聚类配置
    config['cluster_graph'] = {
        "max_cluster_size": 10,
    }

    # 5. 向量库配置
    config['vector_store'] = {
        "default_vector_store": {
            "type": "lancedb",
            "db_uri": str(settings.basic_settings.VS_DIR / "graphrag_lancedb").replace('\\', '/'),
            "overwrite": True,
        }
    }

    # 6. 工作流列表
    config['workflows'] = [
        "load_input_documents",
        "use_document_as_text_unit",
        "create_final_documents",
        "extract_graph",
        "finalize_graph",
        "create_communities",
        "create_community_reports",
        "generate_text_embeddings",
    ]

    with open(settings_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)



def prepare_chunks_to_txt(json_path: Path, output_txt_dir: Path):
    """
    读取 JSON 分块文件，将每个 chunk 的 concat_text 写入独立的 .txt 文件。
    文件名为 chunk_id.txt。
    """
    output_txt_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    if not chunks:
        logger.warning(f"No chunks found in {json_path}")
        return 0

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            logger.warning("Skipping chunk without chunk_id")
            continue
        concat_text = chunk.get("concat_text", "")
        if not concat_text:
            logger.warning(f"Chunk {chunk_id} has empty concat_text, skipping")
            continue
        txt_file = output_txt_dir / f"{chunk_id}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(concat_text)
    logger.info(f"Written {len(chunks)} chunk files to {output_txt_dir}")
    return len(chunks)


def run_graphrag_index(workspace_dir: Path):
    """
    注册工作流，将一个分块作为一个文本单元，不进行切分
    运行 GraphRAG 索引流程
    """
    os.environ['LITELLM_LOG'] = 'DEBUG'

    # 1. 注册自定义工作流
    from graphrag.index.workflows import PipelineFactory
    PipelineFactory.register('use_document_as_text_unit', custom_chunk_workflow)

    # 2. 加载配置为字典，然后创建配置对象
    from graphrag.config.load_config import load_config
    config = load_config(root_dir=workspace_dir)

    # 3. 异步运行索引 (同样使用 build_index)
    import asyncio
    from graphrag.api import build_index

    async def run():
        await build_index(config=config)

    asyncio.run(run())
    logger.info("GraphRAG indexing completed successfully.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ==================== 1. 定义路径 ====================
    chunk_json_path = Path(settings.basic_settings.CHUNKS_DIR) / "semantic_split_b_1_p_90.json"
    mock_chunk_json_path = Path(settings.basic_settings.CHUNKS_DIR) / "semantic_mock.json"
    graphrag_base_dir = Path(settings.basic_settings.SERVER_ROOT) / "graph_rag_model"
    workspace_root = graphrag_base_dir / "workspace"
    input_dir = workspace_root / "input"

    try:
        # ==================== 2. 清理旧工作空间（复用 workspace 结果，从工作流中间环节开始执行时注销掉这一步）====================
        # if workspace_root.exists():
        #     shutil.rmtree(workspace_root)
        #     logger.info(f"Cleaned up previous workspace: {workspace_root}")
        #
        # # ==================== 3. 初始化 GraphRAG（复用 workspace 结果，从工作流中间环节开始执行时注销掉这一步）====================
        # init_cmd = [sys.executable, "-m", "graphrag", "init", "--root", str(workspace_root)]
        # logger.info(f"Initializing GraphRAG workspace: {' '.join(init_cmd)}")
        # subprocess.run(init_cmd, check=True)
        #
        # # ==================== 4. 准备输入数据（复用 workspace 结果，从工作流中间环节开始执行时注销掉这一步） ====================
        # prepare_chunks_to_txt(chunk_json_path, input_dir)
        #
        # ==================== 5. 增量修改 settings.yaml ====================
        update_graphrag_settings(workspace_root)

        # ==================== 6. 执行索引构建 ====================
        run_graphrag_index(workspace_root)

    except Exception as e:
        logger.error(f"Example ingestion failed: {e}")
    logging.basicConfig(level=logging.INFO)
