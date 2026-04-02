from __future__ import annotations

import asyncio, json
import uuid
from typing import AsyncIterable, List, Optional, Literal

from fastapi import Body, Request
from fastapi.concurrency import run_in_threadpool

from sse_starlette.sse import EventSourceResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts.chat import ChatPromptTemplate

from server.kb_service.chromadb_service import SimpleChromaKB
from server import settings
from server.utils import logger


# 用户进入聊天服务界面
"""
注：
毕设专注于 针对对话流问答的精确性，为降低系统的复杂度
1. 不支持与智能问答系统的多轮对话，只关注单个 query 回答的精确性
"""
async def chat_service(
    query: str = Body(..., description="用户输入", examples=["你好"]),
    kb_name: str = Body("", description="知识库名称", examples=["samples"]),
    top_k: int = Body(settings.kb_settings.VECTOR_SEARCH_TOP_K, description="匹配向量数"),
    score_threshold: float = Body(
        settings.kb_settings.SCORE_THRESHOLD,
        description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高",
        ge=0,
        le=2,
    ),
    stream: bool = Body(True, description="流式输出"),
    model: str = Body(settings.api_model_settings.DEFAULT_LLM_MODEL, description="LLM 模型名称"),
    temperature: float = Body(settings.api_model_settings.TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
    max_tokens: Optional[int] = Body(settings.api_model_settings.MAX_TOKENS, description="限制LLM生成Token数量"),
    prompt_name: str = Body("default", description="使用的prompt模板名称(在prompt_settings.yaml中配置)"),
    # 是否直接返回检索结果，不调用LLM生成
    return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
):

    # 1. 初始化知识库服务
    try:
        kb = SimpleChromaKB(kb_name=kb_name)
    except Exception as e:
        return {"code": 404, "msg": f"知识库 {kb_name} 初始化失败: {e}"}

    # 异步迭代器，SSE流式输出核心
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            # 声明使用外部函数的变量
            nonlocal prompt_name, max_tokens

            # 2. 检索文档（放入线程池异步执行，避免阻塞），格式化文档来源
            docs = await run_in_threadpool(kb.search, query=query, top_k=top_k, score_threshold=score_threshold)
            source_documents = []
            for i, doc in enumerate(docs):
                source_info = f"出处 [{i + 1}] {doc.metadata.get('source', '未知来源')}\n\n{doc.page_content}\n\n"
                source_documents.append(source_info)

            # 直接返回检索结果
            if return_direct:
                yield json.dumps({
                    "id": f"chat{uuid.uuid4()}",
                    "model": None,
                    "object": "chat.completion",
                    "content": "",
                    "role": "assistant", 
                    "finish_reason": "stop",
                    "docs": source_documents,
                })
                return


            # 3. 初始化 LLM
            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]

            if max_tokens in [None, 0]:
                max_tokens = settings.api_model_settings.MAX_TOKENS

            # 默认 glm-4-plus
            from langchain_community.chat_models import ChatZhipuAI
            llm = ChatZhipuAI(
                api_key=settings.platform_config.api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                streaming=True,
                callbacks=callbacks,
            )
            # 使用 glm-4-plus
            if model.startswith("glm-4"):
                llm = ChatZhipuAI(
                    api_key=settings.platform_config.api_key,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=True,
                    callbacks=callbacks,
                )


            # 4. 构建提示词（query + page_content + history）
            context = "\n\n".join([doc.page_content for doc in docs])
            question = query
            if len(docs) == 0:
                # 没搜到内容时的兜底模板
                system_prompt = "请直接回答用户的问题。"
                user_content = f"用户问题: {question}"
            else:
                system_prompt = "你是一个基于参考信息的问答助手。请严格根据参考信息回答问题，如果信息不足请直说。"
                user_content = f"参考信息:\n{context}\n\n用户问题: {question}\n\n请用中文回答:"
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_content)
            ])
            chain = chat_prompt | llm


            # 5. 异步执行 LLM 调用
            task = asyncio.create_task(asyncio.wait_for(
                chain.ainvoke({}),
                timeout=30.0
            ))

            # 6. 未找到文档，回答中要添加提示
            if len(source_documents) == 0:
                source_documents.append("<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

            # 7. 流式/非流式输出
            if stream:
                # 第一步：先推送参考文档
                ret = {
                    "id": f"chat{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "content": "",
                    "role": "assistant",
                    "model": model,
                    "docs": source_documents,
                }
                yield json.dumps(ret)

                # 第二步：逐 token 推送 LLM 输出
                async for token in callback.aiter():
                    ret = {
                        "id": f"chat{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "content": token,
                        "role": "assistant",
                        "model": model,
                    }
                    yield json.dumps(ret)
            else:
                # 非流式：拼接所有 token，一次性返回
                answer = ""
                async for token in callback.aiter():
                    answer += token
                ret = {
                    "id": f"chat{uuid.uuid4()}",
                    "object": "chat.completion",
                    "content": answer,
                    "role": "assistant",
                    "model": model,
                    "docs": source_documents,
                }
                yield json.dumps(ret)

            # 等待后台任务完成
            await task

        # 用户中断流式输出
        except asyncio.exceptions.CancelledError:
            logger.warning("streaming progress has been interrupted by user.")
            return
        # 超时异常
        except asyncio.TimeoutError:
            logger.error("LLM调用超时")
            yield json.dumps({"error": "LLM调用超时，请稍后重试"})
            return
        # 全局异常捕获
        except Exception as e:
            logger.error(f"error in knowledge chat: {e}")
            yield json.dumps({"error": str(e)})
            return

    # 8. 接口返回值
    if stream:  # 流式：返回SSE响应对象
        return EventSourceResponse(knowledge_base_chat_iterator())
    else:   # 非流式：直接返回迭代器第一个结果
        return await knowledge_base_chat_iterator().__anext__()