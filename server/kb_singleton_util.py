import json
import os

# 禁用 ChromaDB 遥测
os.environ["CHROMA_DISABLE_TELEMETRY"] = "1"

from typing import List
from functools import lru_cache


import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from server import settings

"""
Manage Chromadb Singleton Resources
"""


class ChromaResourceManager:
    _client = None
    _embeddings = {}

    @classmethod
    def get_client(cls, path: str):
        if cls._client is None:
            cls._client = chromadb.PersistentClient(path=path)
            print(f"[kb 资源管理器] 首次加载全局单例 Chroma 客户端，路径: {path}")
        return cls._client

    @classmethod
    def get_embeddings(cls, model_name: str, api_key: str = None, base_url: str = None):
        if model_name not in cls._embeddings:
            try:
                # 支持
                # - embedding-3
                # - text-embedding-3-small
                # - BAAI/bge-m3 (Local)
                if "embedding-3" in model_name or "text-embedding" in model_name:
                    cls._embeddings[model_name] = OpenAIEmbeddings(
                        model=model_name,
                        api_key=api_key,
                        base_url=base_url
                    )
                    print(f"[kb 资源管理器] 成功初始化远程模型: {model_name}")
                elif "bge-m3" in model_name.lower():
                    cls._embeddings[model_name] = HuggingFaceBgeEmbeddings(
                        model_name=model_name,
                        model_kwargs={'device': 'cpu'}, 
                        encode_kwargs={'normalize_embeddings': True}
                    )
                    print(f"[kb 资源管理器] 成功初始化本地模型: {model_name}")
                else:
                    print(f"[kb 资源管理器] 仅支持 embedding-3、text-embedding-3-small 和 BGE-M3 模型")
            except Exception as e:
                print(f"[kb 资源管理器] 初始化模型 {model_name} 失败: {e}")
        return cls._embeddings[model_name]


"""
- embedding_model
"""


class SimpleChromaKB:
    kb_name: str
    vs_path: str

    embedding_model_name: str
    embedding_function: Embeddings

    client: chromadb.PersistentClient  # Chroma Client
    template: Chroma  # LangChain Chroma Wrapper

    def __init__(
            self,
            kb_name: str,
            vs_path: str = None,
            embedding_model_name: str = None):

        # 1. Path
        self.kb_name = kb_name
        self.vs_path = vs_path or str(settings.basic_settings.VS_PATH)
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)

        # 2. Get Singleton Client
        self.client = ChromaResourceManager.get_client(self.vs_path)

        # 3. Get Singleton Embeddings
        self.embedding_model_name = embedding_model_name or settings.api_model_settings.DEFAULT_EMBEDDING_MODEL
        
        # Get model configuration
        model_config = settings.api_model_settings.MODELS.get(self.embedding_model_name)
        platform_name = model_config.platform_name if model_config else "zhipuai"
        
        # Get platform configuration
        target_key = None
        target_url = None
        for platform in settings.api_model_settings.MODEL_PLATFORMS:
            if platform.platform_name == platform_name:
                target_key = platform.api_key
                target_url = platform.api_embedding_base_url
                break
        
        self.embedding_function = ChromaResourceManager.get_embeddings(
            model_name=self.embedding_model_name,
            api_key=target_key,
            base_url=target_url
        )

        # 4. Initialize LangChain Chroma Wrapper
        self.template = Chroma(
            client=self.client,
            collection_name=self.kb_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, documents: List[Document]):
        if not documents:
            return []
        ids = self.template.add_documents(documents)
        return ids

    def delete_collection(self):
        self.client.delete_collection(self.kb_name)
        # Reinitialize the template to create a new collection
        self.template = Chroma(
            client=self.client,
            collection_name=self.kb_name,
            embedding_function=self.embedding_function,
        )
        print(f"[kb 资源管理器] 已重新初始化集合: {self.kb_name}")

    def search(self, query: str, top_k: int = 5):
        results = self.template.similarity_search_with_score(query, k=top_k)

        formatted_results = []
        for doc, score in results:
            source_chunk_ids = json.loads(doc.metadata.get('source_chunk_ids', '[]'))
            chunk_id = doc.metadata.get('chunk_id', '')
            if chunk_id:
                formatted_results.append({
                    'text': doc.page_content,
                    'chunk_id': chunk_id,
                    'score': score
                })
            else:
                formatted_results.append({
                    'summary': doc.page_content,
                    'chunk_ids': source_chunk_ids,
                    'score': score
                })
        return formatted_results


def get_kb(
        kb_name: str,
        vs_path: str = None,
        embedding_model_name: str = None
) -> SimpleChromaKB:
    return SimpleChromaKB(
        kb_name=kb_name,
        vs_path=vs_path,
        embedding_model_name=embedding_model_name
    )


# Warm Up For The First Time Being Imported
print(">>> 正在初始化全局知识库基础资源...")
ChromaResourceManager.get_client(str(settings.basic_settings.VS_PATH))
ChromaResourceManager.get_embeddings(
    model_name=settings.api_model_settings.DEFAULT_EMBEDDING_MODEL,
)
print(">>> 全局知识库基础资源初始化完毕。")

if __name__ == "__main__":
    pass