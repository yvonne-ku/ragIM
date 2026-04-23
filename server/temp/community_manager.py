import networkx as nx
from typing import List, Dict, Any, Tuple
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from server import settings
from resources.others.utils import logger

class CommunityManager:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.api_model_settings.DEFAULT_LLM_MODEL
        self.llm = ChatZhipuAI(
            api_key=settings.platform_config.api_key,
            model=self.model_name,
            temperature=0.3,
        )
        self.graph = nx.Graph()
        
        # 社区摘要提示词
        self.summary_prompt = ChatPromptTemplate.from_template("""
你是一个专业的社区摘要生成助手。请根据以下社区内的实体及其关系，生成一个简洁、专业的摘要。

社区实体：
{entities}

社区关系：
{relationships}

摘要规则：
1. 包含社区的核心主题和主要内容。
2. 保持摘要在 200 字以内。
3. 输出摘要内容。
""")

    def build_graph(self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]):
        """构建 NetworkX 图"""
        for ent in entities:
            self.graph.add_node(ent["name"], **ent)
        
        for rel in relationships:
            u, v = rel["source"], rel["target"]
            # 聚合权重：如果已有边，则增加权重
            if self.graph.has_edge(u, v):
                self.graph[u][v]["weight"] += 1
                self.graph[u][v]["msg_ids"].extend(rel.get("msg_ids", []))
                self.graph[u][v]["msg_ids"] = list(set(self.graph[u][v]["msg_ids"]))
            else:
                self.graph.add_edge(u, v, weight=1, relation=rel["relation"], description=rel["description"], msg_ids=rel.get("msg_ids", []))

    def detect_communities_leiden(self) -> List[List[str]]:
        """
        Leiden 算法实现。
        由于 leidenalg 可能未安装，这里使用 NetworkX 的 Louvain 算法作为 fallback。
        实际使用中建议安装 `pip install leidenalg igraph`
        """
        try:
            import leidenalg
            import igraph as ig
            
            # 转换为 igraph
            g_ig = ig.Graph.from_networkx(self.graph)
            # 运行 Leiden
            partition = leidenalg.find_partition(g_ig, leidenalg.ModularityVertexPartition)
            communities = []
            for cluster in partition:
                nodes = [g_ig.vs[idx]["_nx_name"] for idx in cluster]
                communities.append(nodes)
            return communities
        except ImportError:
            logger.warning("leidenalg 未安装，回退到 louvain 算法。建议运行: pip install leidenalg igraph")
            from networkx.community import louvain_communities
            return [list(c) for c in louvain_communities(self.graph)]

    def generate_community_summary(self, nodes: List[str]) -> str:
        """为特定社区生成摘要"""
        # 提取社区内的实体和关系
        entities_info = []
        for node in nodes:
            data = self.graph.nodes[node]
            entities_info.append(f"实体: {node}, 类型: {data.get('type', '未知')}, 描述: {data.get('description', '')}")
            
        relationships_info = []
        subgraph = self.graph.subgraph(nodes)
        for u, v, data in subgraph.edges(data=True):
            relationships_info.append(f"{u} --[{data.get('relation', '关系')}]--> {v}: {data.get('description', '')}")
            
        # 调用 LLM 生成摘要
        chain = self.summary_prompt | self.llm
        response = chain.invoke({
            "entities": "\n".join(entities_info),
            "relationships": "\n".join(relationships_info)
        })
        return response.content

    def build_hierarchical_communities(self) -> List[Dict[str, Any]]:
        """构建多层次社区及摘要"""
        # 这里简化为一层级社区。实际 GraphRAG 可能会递归进行。
        communities = self.detect_communities_leiden()
        hierarchical_data = []
        
        for i, comm_nodes in enumerate(communities):
            summary = self.generate_community_summary(comm_nodes)
            # 收集社区关联的所有 msg_ids
            all_msg_ids = []
            subgraph = self.graph.subgraph(comm_nodes)
            for _, _, data in subgraph.edges(data=True):
                all_msg_ids.extend(data.get("msg_ids", []))
            
            hierarchical_data.append({
                "community_id": i,
                "nodes": comm_nodes,
                "summary": summary,
                "msg_ids": list(set(all_msg_ids))
            })
            
        return hierarchical_data
