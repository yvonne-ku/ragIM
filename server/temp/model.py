import os
import json
import networkx as nx
from typing import List, Dict, Any
from .entity_extractor import EntityExtractor
from .community_manager import CommunityManager
from .retriever import DualPathRetriever
from resources.others.utils import logger

class GraphHierarchyModel:
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.join(os.getcwd(), "storage", "graph_rag")
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
            
        self.extractor = EntityExtractor()
        self.manager = CommunityManager()
        self.retriever = None
        self.hierarchical_communities = []

    def ingest(self, json_data: Dict[str, Any]):
        """
        处理数据：实体抽取 -> 构建图 -> 社区划分 -> 摘要生成
        """
        # 1. 抽取实体与关系
        chunks = json_data.get("chunks", [])
        extracted = self.extractor.extract_from_chunks(chunks)
        
        # 2. 构建图
        self.manager.build_graph(extracted["entities"], extracted["relationships"])
        
        # 3. 社区划分与摘要生成
        self.hierarchical_communities = self.manager.build_hierarchical_communities()
        
        # 4. 初始化检索器
        self.retriever = DualPathRetriever(self.manager.graph, self.hierarchical_communities)
        
        # 5. 保存状态
        self.save()
        logger.info(f"GraphHierarchyModel 摄取完成。实体数: {self.manager.graph.number_of_nodes()}, 社区数: {len(self.hierarchical_communities)}")

    def search(self, query: str, top_k: int = 3, bfs_depth: int = 2) -> List[str]:
        """
        双路径检索
        """
        if self.retriever is None:
            self.load()
            
        return self.retriever.retrieve(query, top_k=top_k, bfs_depth=bfs_depth)

    def save(self):
        """保存图和社区信息"""
        # 保存 NetworkX 图 (GML 格式)
        nx.write_gml(self.manager.graph, os.path.join(self.storage_path, "graph.gml"))
        
        # 保存社区摘要
        with open(os.path.join(self.storage_path, "communities.json"), "w", encoding="utf-8") as f:
            json.dump(self.hierarchical_communities, f, ensure_ascii=False, indent=2)

    def load(self):
        """加载图和社区信息"""
        graph_path = os.path.join(self.storage_path, "graph.gml")
        communities_path = os.path.join(self.storage_path, "communities.json")
        
        if os.path.exists(graph_path):
            self.manager.graph = nx.read_gml(graph_path)
            
        if os.path.exists(communities_path):
            with open(communities_path, "r", encoding="utf-8") as f:
                self.hierarchical_communities = json.load(f)
                
        if self.manager.graph.number_of_nodes() > 0:
            self.retriever = DualPathRetriever(self.manager.graph, self.hierarchical_communities)
            logger.info("GraphHierarchyModel 加载成功。")
        else:
            logger.warning("GraphHierarchyModel 加载失败，图为空。")
