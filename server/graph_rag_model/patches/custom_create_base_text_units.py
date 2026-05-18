# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import logging
from typing import cast

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage
from graphrag.index.operations.chunk_text.strategies import get_encoding_fn

logger = logging.getLogger(__name__)


async def run_workflow(
        config: GraphRagConfig,
        context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform base text_units by treating each doc as one unit."""
    logger.info("Workflow started: use_document_as_text_unit")

    # 1. 加载由 load_input_documents 产生的 documents 表
    documents = await load_table_from_storage("documents", context.output_storage)

    # 2. 调用修改后的处理逻辑
    output = create_base_text_units_as_single_unit(
        documents,
        config.chunks.encoding_model
    )

    # 3. 写入 text_units 供后续 extract_graph 使用
    await write_table_to_storage(output, "text_units", context.output_storage)

    logger.info("Workflow completed: use_document_as_text_unit")
    return WorkflowFunctionOutput(result=output)


def create_base_text_units_as_single_unit(
        documents: pd.DataFrame,
        encoding_model: str,
) -> pd.DataFrame:
    """直接将每个文档转换为单个 text_unit，不进行任何切片。"""

    # 获取编码函数以计算 token 数量（下游流程需要 n_tokens 字段）
    encode, _ = get_encoding_fn(encoding_model)

    # 构造下游预期的 DataFrame 结构
    # GraphRAG 下游期望的 text_units 表至少包含: [id, text, document_ids, n_tokens]

    res_data = []
    for _, row in documents.iterrows():
        doc_text = str(row["text"])
        doc_id = str(row["id"])

        # 计算当前文档的 token 数
        tokens = encode(doc_text)
        n_tokens = len(tokens)

        # 构造单元数据
        unit_row = {
            "text": doc_text,
            "document_ids": [doc_id],  # 必须是列表，因为一个 unit 可能对应多个 doc（虽然这里是 1:1）
            "n_tokens": n_tokens,
        }

        # 为这个 unit 生成唯一的哈希 ID
        unit_row["id"] = gen_sha512_hash(unit_row, ["text"])

        res_data.append(unit_row)

    output = pd.DataFrame(res_data)

    return output.reset_index(drop=True)