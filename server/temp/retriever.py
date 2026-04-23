import json
import networkx as nx
from typing import List, Dict, Any, Set
from langchain_core.embeddings import Embeddings
from server.kb_singleton_util import ChromaResourceManager
from server import settings
from .entity_extractor import EntityExtractor
from resources.others.utils import logger
import numpy as np

class DualPathRetriever:
    def __init__(self, 
                 graph: nx.Graph, 
                 hierarchical_communities: List[Dict[str, Any]], 
                 embedding_model: Embeddings = None):
        self.graph = graph
        self.communities = hierarchical_communities
        # 默认使用 BGE-M3
        self.embedding_function = embedding_model or ChromaResourceManager.get_embeddings(
            settings.api_model_settings.DEFAULT_EMBEDDING_MODEL
        )
        self.entity_extractor = EntityExtractor()

    def cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def retrieve_path_1(self, query: str, top_k: int = 3) -> Set[str]:
        """路径 1：与层次化摘要进行相似度比较"""
        query_embedding = self.embedding_function.embed_query(query)
        
        scores = []
        for comm in self.communities:
            summary_embedding = self.embedding_function.embed_query(comm["summary"])
            score = self.cosine_similarity(query_embedding, summary_embedding)
            scores.append((score, comm))
            
        # 按得分排序并取前 top_k
        scores.sort(key=lambda x: x[0], reverse=True)
        top_communities = scores[:top_k]
        
        msg_id_set = set()
        for _, comm in top_communities:
            msg_id_set.update(comm.get("msg_ids", []))
            
        return msg_id_set

    def retrieve_path_2(self, query: str, bfs_depth: int = 2) -> Set[str]:
        """路径 2：查询抽取实体 -> 字符串匹配 -> BFS -> msg_id_set"""
        # 1. 抽取查询中的实体
        extracted = self.entity_extractor.extract(query)
        query_entities = [e["name"] for e in extracted.get("entities", [])]
        
        if not query_entities:
            # 尝试最简单的词匹配
            query_entities = [word for word in query.split() if len(word) > 1]
            
        # 2. 字符串匹配图中的实体
        matched_nodes = []
        graph_nodes = list(self.graph.nodes())
        for q_ent in query_entities:
            for g_node in graph_nodes:
                if q_ent in g_node or g_node in q_ent:
                    matched_nodes.append(g_node)
        
        matched_nodes = list(set(matched_nodes))
        
        # 3. 从匹配到的实体开始 BFS
        visited_nodes = set()
        msg_id_set = set()
        
        for start_node in matched_nodes:
            # 使用 NetworkX 的 BFS
            bfs_edges = list(nx.bfs_edges(self.graph, start_node, depth_limit=bfs_depth))
            current_nodes = {start_node}
            for u, v in bfs_edges:
                current_nodes.add(u)
                current_nodes.add(v)
            
            # 收集关联的 msg_ids
            subgraph = self.graph.subgraph(current_nodes)
            for _, _, data in subgraph.edges(data=True):
                msg_id_set.update(data.get("msg_ids", []))
                
        return msg_id_set

    def retrieve(self, query: str, top_k: int = 3, bfs_depth: int = 2) -> List[str]:
        """双路径检索汇总"""
        p1_msg_ids = self.retrieve_path_1(query, top_k=top_k)
        p2_msg_ids = self.retrieve_path_2(query, bfs_depth=bfs_depth)
        
        # 合并结果
        combined_msg_ids = list(p1_msg_ids.union(p2_msg_ids))
        logger.info(f"检索到 {len(combined_msg_ids)} 个消息记录。路径1: {len(p1_msg_ids)}, 路径2: {len(p2_msg_ids)}")
        
        return combined_msg_ids
