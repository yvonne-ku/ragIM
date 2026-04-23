from __future__ import annotations

import asyncio
import json
import uuid
from typing import AsyncIterable, Optional

from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatZhipuAI

from server.kb_service.chromadb_service import get_kb
from server import settings
from resources.others.utils import logger


async def chat_service(
        query: str = Body(..., description="用户输入", examples=["你好"]),
        kb_name: str = Body("", description="知识库名称", examples=["samples"]),
        top_k: int = Body(settings.kb_settings.VECTOR_SEARCH_TOP_K, description="匹配向量数"),
        score_threshold: float = Body(
            settings.kb_settings.SCORE_THRESHOLD,
            description="知识库匹配相关度阈值",
            ge=0,
            le=2,
        ),
        stream: bool = Body(True, description="流式输出"),
        model: str = Body(settings.api_model_settings.DEFAULT_LLM_MODEL, description="LLM 模型名称"),
        temperature: float = Body(settings.api_model_settings.TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
        max_tokens: Optional[int] = Body(settings.api_model_settings.MAX_TOKENS, description="限制LLM生成Token数量"),
        prompt_name: str = Body("default", description="使用的prompt模板名称"),
        return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
):
    # 1. 初始化知识库
    try:
        kb = get_kb(kb_name=kb_name)
    except Exception as e:
        return {"code": 404, "msg": f"知识库 {kb_name} 初始化失败: {e}"}

    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            # 2. 检索文档
            docs = await run_in_threadpool(kb.search, query=query, top_k=top_k, score_threshold=score_threshold)

            source_documents = []
            for i, doc in enumerate(docs):
                source_info = f"出处 [{i + 1}] {doc.metadata.get('source', '未知来源')}\n\n{doc.page_content}\n\n"
                source_documents.append(source_info)

            # 未找到文档提示
            if len(source_documents) == 0:
                source_documents.append("<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

            # 直接返回检索结果逻辑
            if return_direct:
                yield json.dumps({
                    "id": f"chat{uuid.uuid4()}",
                    "model": model,
                    "object": "chat.completion",
                    "content": "",
                    "role": "assistant",
                    "finish_reason": "stop",
                    "docs": source_documents,
                })
                return

            # 3. 初始化 LLM (显式加上 base_url)
            llm = ChatZhipuAI(
                api_key=settings.platform_config.api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens or settings.api_model_settings.MAX_TOKENS,
                streaming=True,
            )

            # 4. 构建提示词
            context = "\n\n".join([doc.page_content for doc in docs])
            cur_prompt_key = prompt_name if len(docs) > 0 else "empty"
            raw_template = settings.prompt_settings.rag.get(
                cur_prompt_key,
                settings.prompt_settings.rag.get("default"),
            )
            # 替换模板占位符
            template_str = raw_template.replace("{{context}}", "{context}").replace("{{question}}", "{question}")
            chat_prompt = ChatPromptTemplate.from_template(template_str)

            # 5. 组合 Chain
            chain = chat_prompt | llm

            # 6. 执行与输出
            if stream:
                # 第一步：先推送参考文档 (Chunk 模式)
                yield json.dumps({
                    "id": f"chat{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "content": "",
                    "role": "assistant",
                    "model": model,
                    "docs": source_documents,
                })

                # 第二步：使用 astream 直接异步迭代获取 token
                async for chunk in chain.astream({"context": context, "question": query}):
                    yield json.dumps({
                        "id": f"chat{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "content": chunk.content,  # ChatZhipuAI 返回的是消息对象，取 .content
                        "role": "assistant",
                        "model": model,
                    })
            else:
                # 非流式：直接调用 ainvoke
                response = await chain.ainvoke({"context": context, "question": query})
                yield json.dumps({
                    "id": f"chat{uuid.uuid4()}",
                    "object": "chat.completion",
                    "content": response.content,
                    "role": "assistant",
                    "model": model,
                    "docs": source_documents,
                })

        except asyncio.exceptions.CancelledError:
            logger.warning("Streaming interrupted by user.")
        except Exception as e:
            logger.error(f"Error in knowledge chat: {e}")
            yield json.dumps({"error": str(e)})

    # 7. 返回接口响应
    if stream:
        return EventSourceResponse(knowledge_base_chat_iterator())
    else:
        # 非流式直接运行并返回
        return await knowledge_base_chat_iterator().__anext__()