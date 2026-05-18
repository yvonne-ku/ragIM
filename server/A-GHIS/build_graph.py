import json
import os
import pickle
from typing import List, Dict, Any, Tuple, Set, Optional

import networkx as nx

from server import settings

EXTRACT_ENTITY_PROMPT = settings.prompt_settings.entity_extraction.get("loose_extraction_for_more_entities_and_relations")

def build_graph_from_chunks(chunks_data: List[Dict[str, Any]]) -> nx.Graph:
    """
    全局图构建函数
    核心改进：
    1. 实体名称归一化，减少重复节点
    2. 增加同chunk内实体共现边，解决图碎片化问题
    3. 显式关系权重高于共现关系，保留知识可靠性
    """

    def normalize_entity_name(name: str) -> str:
        """简单实体归一化：去除首尾空格、统一大小写（可根据需要扩展）"""
        if not name:
            return ""
        return name.strip().lower()

    def extract_entities_relations_from_chunk(
            concat_text: str,
            model: str,
            chunk_id: int
    ) -> Tuple[List[str], List[Dict]]:
        """
        调用LLM提取实体和关系（保留原有逻辑，增加空值过滤）
        """
        print(f"   [Chunk {chunk_id}] 开始调用LLM提取实体关系...")
        # 1. Prepare prompt
        prompt = EXTRACT_ENTITY_PROMPT.format(conversation=concat_text)
        llm_messages = [{"role": "user", "content": prompt}]

        # 2. Call LLM API with timeout and retries
        from langchain_openai import ChatOpenAI
        platform_config = settings.api_model_settings.MODEL_PLATFORMS.get("zhipuai")
        model_config = settings.api_model_settings.MODELS.get(model)
        llm = ChatOpenAI(
            model=model,
            api_key=platform_config.api_key if platform_config else None,
            base_url=platform_config.base_url if platform_config else None,
            temperature=model_config.temperature if model_config else None,
            timeout=60,
            max_retries=2
        )
        try:
            response = llm.invoke(llm_messages)
            response_text = response.content
        except Exception as e:
            print(f"   [Chunk {chunk_id}] LLM调用失败: {e}")
            return [], []

        # 3. Parse JSON from response
        try:
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No valid JSON found")
            json_str = response_text[start:end]
            data = json.loads(json_str)
            # 过滤空实体和空关系
            entities = [e.strip() for e in data.get("entities", []) if e and e.strip()]
            relations = [
                r for r in data.get("relations", [])
                if r.get("source") and r.get("source").strip()
                   and r.get("target") and r.get("target").strip()
            ]
            print(f"   [Chunk {chunk_id}] 提取完成: {len(entities)} 个实体, {len(relations)} 个关系")
            return entities, relations
        except Exception as e:
            print(f"   [Chunk {chunk_id}] JSON解析失败: {e}\nRaw output: {response_text[:200]}...")
            return [], []

    G = nx.Graph()
    total = len(chunks_data)
    print(f"开始构建图，共 {total} 个chunk...")

    for idx, chunk in enumerate(chunks_data):
        chunk_id = chunk["chunk_id"]
        concat_text = chunk["concat_text"]
        print(f"处理 chunk {idx + 1}/{total} (ID={chunk_id})")

        model = "glm-4"
        entities, relations = extract_entities_relations_from_chunk(concat_text, model, chunk_id)

        # 跳过没有实体的chunk
        if not entities:
            print(f"   [Chunk {chunk_id}] 未提取到任何实体，跳过")
            continue

        # 2. 构建实体节点（增加归一化）
        entity_id_map = {}  # 原始名称 -> 归一化后的节点ID
        for ent_name in entities:
            normalized_name = normalize_entity_name(ent_name)
            ent_id = f"entity::{normalized_name}"
            entity_id_map[ent_name] = ent_id

            if ent_id not in G:
                G.add_node(
                    ent_id,
                    type="entity",
                    name=ent_name,
                    normalized_name=normalized_name,
                    source_ids={chunk_id},
                    occurrence_count=1
                )
            else:
                G.nodes[ent_id]["source_ids"].add(chunk_id)
                G.nodes[ent_id]["occurrence_count"] += 1

        # 3. 构建显式关系边（权重=2，优先级高于共现边）
        for rel in relations:
            src_name = rel.get("source", "").strip()
            tgt_name = rel.get("target", "").strip()
            desc = rel.get("description", "").strip()

            src_id = entity_id_map.get(src_name)
            tgt_id = entity_id_map.get(tgt_name)

            if not src_id or not tgt_id or src_id == tgt_id:
                continue

            if G.has_edge(src_id, tgt_id):
                edge_data = G.edges[src_id, tgt_id]
                edge_data["source_ids"].add(chunk_id)
                # 显式关系每次加2，共现关系每次加1
                edge_data["weight"] += 2
                if desc and desc not in edge_data["relation"]:
                    edge_data["relation"] += f"；{desc}"
            else:
                G.add_edge(
                    src_id, tgt_id,
                    relation=desc,
                    source_ids={chunk_id},
                    weight=2,
                    edge_type="explicit"  # 标记为显式关系
                )

        # 4. 构建同chunk实体共现边
        # 同一个chunk中出现的所有实体两两建立共现边，权重=1
        entity_ids = list(entity_id_map.values())
        if len(entity_ids) >= 2:
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    src_id = entity_ids[i]
                    tgt_id = entity_ids[j]

                    if src_id == tgt_id:
                        continue

                    if G.has_edge(src_id, tgt_id):
                        # 如果已经有边（显式或共现），权重+1
                        G.edges[src_id, tgt_id]["weight"] += 1
                        G.edges[src_id, tgt_id]["source_ids"].add(chunk_id)
                    else:
                        # 新增共现边
                        G.add_edge(
                            src_id, tgt_id,
                            relation="共现于同一chunk",
                            source_ids={chunk_id},
                            weight=1,
                            edge_type="co-occurrence"  # 标记为共现关系
                        )

        # 进度打印
        if (idx + 1) % 20 == 0:
            print(
                f"   [进度] 已处理 {idx + 1}/{total} 个chunk, 当前图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

    # 最终统计
    print(f"\n所有chunk处理完成！")
    print(f"最终图统计: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    print(f"孤立点数量: {len(list(nx.isolates(G)))}")
    print(f"连通分量数量: {len(list(nx.connected_components(G)))}")
    print(f"最大连通分量大小: {max(len(c) for c in nx.connected_components(G))}")

    return G


if __name__ == "__main__":

    json_path = settings.basic_settings.CHUNKS_DIR / "semantic_split_b_1_p_90.json"
    output_dir = settings.basic_settings.SERVER_ROOT / "A-GHIS" / "workspace"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 chunk
    # max_count 截断加载数量，可按需进行试验
    max_chunks = None
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks_list = json.load(f)["chunks"]
    if max_chunks is not None:
        chunks_list = chunks_list[:max_chunks]
    print(f"Loaded {len(chunks_list)} chunks (limit={max_chunks if max_chunks else 'all'}).")

    # 2. 构建图
    G = build_graph_from_chunks(chunks_list)
    graph_path = os.path.join(output_dir, "graph.pkl")
    with open(graph_path, 'wb') as f:
        pickle.dump(G, f, protocol=4)
    print(f"图已保存到 {graph_path}")

