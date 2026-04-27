import os
import re
import json
import pickle
from typing import List, Dict, Any, Set
from collections import defaultdict

from server import settings
from server.kb_singleton_util import get_kb
from server.evaluate_utils import evaluate_retrieval

K_RRF = 60
WEIGHT_SUMMARY = 0.4  # 社区摘要权重 (发现大背景)
WEIGHT_EXACT = 0.4  # 字符精确匹配权重 (硬锚点)
WEIGHT_VECTOR = 0.2  # 实体名向量匹配权重 (语义补偿)
EXTRACT_KEYWORDS_PROMPT = settings.prompt_settings.extract_keywords["default"]


def extract_query_keywords(query: str) -> List[str]:
    """
    从查询中提取关键词
    """
    # 1. Prompt
    prompt = EXTRACT_KEYWORDS_PROMPT.format(query=query)
    messages = [{"role": "user", "content": prompt}]

    # 2. Call LLM
    from langchain_openai import ChatOpenAI
    platform_config = settings.api_model_settings.MODEL_PLATFORMS.get("deepseek")
    model = settings.api_model_settings.DEFAULT_EXTRACT_KEYWORDS_MODEL
    model_config = settings.api_model_settings.MODELS.get(model)
    llm = ChatOpenAI(
        model=model,
        api_key=platform_config.api_key if platform_config else None,
        base_url=platform_config.base_url if platform_config else None,
        temperature=model_config.temperature if model_config else None,
    )
    response = llm.invoke(messages)
    raw_content = response.content.strip()

    # 3. Clean And Parse Json String
    try:
        clean_json = re.sub(r'^```json\s*|```$', '', raw_content, flags=re.MULTILINE).strip()
        data = json.loads(clean_json)
        return data.get("keywords", [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error parsing keywords JSON: {e}. Raw content: {raw_content}")
        return [raw_content] if raw_content else []


class GraphHybridRetriever:
    """
    Graph-based Hybrid Retriever for Theme Loop-back discovery.
    Routes: Community Summary (Vector) + Entity Name (Vector) + Entity Name (Exact Str Match)
    """
    def __init__(self,
                 kb_summary_name: str,
                 kb_entity_name: str,
                 graph_pkl_path: str,
                 top_k_comm: int,
                 top_k: int,
                 ):
        self.graph_pkl_path = graph_pkl_path
        self.kb_summary = get_kb(kb_name=kb_summary_name)
        self.kb_entity = get_kb(kb_name=kb_entity_name)
        self.top_k_comm = top_k_comm
        self.top_k = top_k
        self.all_entity_names = self._load_entity_names()

    def _load_entity_names(self) -> Set[str]:
        with open(self.graph_pkl_path, 'rb') as f:
            G = pickle.load(f)
            entity_names = set(G.nodes.get('name', []))
        return set(entity_names)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        combined_results = defaultdict(float)

        # 1. Community Summary
        summary_results = self.kb_summary.search(query, top_k=self.top_k_comm)
        for i, comm in enumerate(summary_results):
            chunk_ids = json.loads(comm.get('metadata', {}).get('source_chunk_ids', '[]'))
            rank = i + 1
            score_to_add = WEIGHT_SUMMARY * (1.0 / (K_RRF + rank))
            for chunk_id in chunk_ids:
                combined_results[chunk_id] += score_to_add

        # 2. Exact Entity Name
        # 3. Entity Vector
        keywords = extract_query_keywords(query)
        exact_matched_entities = []
        for kw in keywords:
            if kw in self.all_entity_names:
                exact_matched_entities.append(kw)

        for i, entity_name in enumerate(exact_matched_entities):
            # 与关键字最相似的实体
            entity_doc = self.kb_entity.search(entity_name, top_k=1)[0]
            source_chunks = json.loads(entity_doc.get('metadata', {}).get('source_chunks', '[]'))

            # 计算得分
            if entity_doc.get('metadata', {}).get('entity_name').lower() == entity_name.lower():
                score_to_add = WEIGHT_EXACT * (1.0 / (K_RRF + 1))   # 精确匹配
            else:
                score_to_add = WEIGHT_VECTOR * (1.0 / (K_RRF + 1))  # 非精确匹配
            # 更新 chunk 得分
            for chunk_id in source_chunks:
                combined_results[chunk_id] += score_to_add

        # 4. Sorted Dict Results
        sorted_data = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        return [{"chunk_id": cid, "score": scr} for cid, scr in sorted_data]


def run_evaluation(
        kb_summary_name: str,
        kb_entity_name: str,
        graph_pkl_path: str,
        top_k_comm: int = 3,
        top_k: int = 5,
):
    # 1. Ensure Graph
    if not os.path.exists(graph_pkl_path):
        print(f"Error: graph file {graph_pkl_path} not found. Run graph building first.")
        return None

    # 2. Initialize retriever
    retriever = GraphHybridRetriever(kb_summary_name, kb_entity_name, graph_pkl_path, top_k_comm, top_k)

    # 3. Evaluate retrieval
    return evaluate_retrieval(retriever)


if __name__ == "__main__":
    # model_instance 应该是你封装好的 LLM 接口
    run_evaluation("kb_ibm_graph_summaries", "kb_ibm_graph_entities", None)