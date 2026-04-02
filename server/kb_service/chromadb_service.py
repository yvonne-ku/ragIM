import os
from typing import List
from langchain.docstore.document import Document
from langchain_chroma import Chroma
import chromadb
from server import settings

class SimpleChromaKB:
    kb_name: str       # KB_ROOT_PATH/kb_name/相关文件
    vs_path: str       # 向量文件存储路径
    
    embedding_model_name: str                       # 嵌入模型名称
    embedding_function: chromadb.EmbeddingFunction  # 嵌入函数

    client: chromadb.PersistentClient       # Chroma 客户端
    template: Chroma                        # LangChain 包装器

    def __init__(
            self, 
            kb_name: str, 
            vs_path: str = None, 
            embedding_model_name: str = "bge-large-zh-v1.5"):  # 使用默认嵌入模型
        
        # 1. 处理 vs 地址
        if vs_path is None:
            vs_path = os.path.join(settings.basic_settings.KB_ROOT, kb_name, "vector_store")
        self.vs_path = vs_path
        self.kb_name = kb_name
        if not os.path.exists(self.vs_path):
            os.makedirs(self.vs_path)

        # 2. 处理嵌入模型，使用本地模型避免API依赖
        self.embedding_model_name = embedding_model_name
        try:
            # 尝试使用智谱AI模型（需要API密钥）
            from langchain_openai import OpenAIEmbeddings
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv("ZHIPUAI_API_KEY")
            if api_key:
                self.embedding_function = OpenAIEmbeddings(
                    model=self.embedding_model_name, 
                    api_key=api_key,
                    base_url="https://open.bigmodel.cn/api/paas/v4")    # 指向智谱的 API 地址
            else:
                raise ImportError("未设置ZHIPUAI_API_KEY")
        except Exception as e:
            print(f"使用智谱AI模型时出错: {e}")
            # 回退到本地模型
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
            print("使用本地嵌入模型（BAAI/bge-small-zh-v1.5）")

        # 3. 初始化 Chroma 客户端和 LangChain 提供的包装器
        self.client = chromadb.PersistentClient(path=self.vs_path)        
        self.template = Chroma(
            client=self.client,
            collection_name=self.kb_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(self, documents: List[Document]):
        """
        将文档切片存入向量数据库
        """
        if not documents:
            return []
                
        ids = self.template.add_documents(documents)
        return ids

    def add_files(self, file_path: str, loader_kwargs: dict = {}):
        """
        将原始数据处理后存入向量数据库
        通过 KnowledgeFile 处理文件并将其存入向量数据库
        """
        from server.file_service.file_service import KnowledgeFile
        
        try:
            # 1. 创建 KnowledgeFile 实例
            kb_file = KnowledgeFile(
                file_path=file_path,
                kb_name=self.kb_name,
                loader_kwargs=loader_kwargs
            )
            
            # 2. files 转化为 docs
            docs = kb_file.file2docs()

            # 3. 切分 docs
            texts = kb_file.docs2texts(docs)    
            
            # 4. 将切分后的文档片段添加到向量数据库中
            if texts:
                return self.add_documents(docs)
            else:
                print(f"文件 {file_path} 处理后未产生任何文档片段。")
                return []
        except Exception as e:
            print(f"添加文件 {file_path} 到知识库 {self.kb_name} 时出错: {e}")
            return []

    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        执行相似度搜索
        """
        docs = self.template.similarity_search(query, k=top_k)
        return docs

    def delete_collection(self):
        """删除当前知识库集合"""
        self.client.delete_collection(self.kb_name)


if __name__ == "__main__":
    
    # 初始化知识库
    my_kb = SimpleChromaKB(
        kb_name="test_kb",
        embedding_model_name="embedding-2"
    )
    
    # 准备几个文档对象 (你可以用你之前学到的 loader 和 splitter 生成这些对象)
    # 这里的测试直接用 Document，在 file_service 中已经测试过 loader 和 splitter
    test_docs = [
        Document(page_content="张三是一个爱学习的大学生。", metadata={"source": "test.txt"}),
        Document(page_content="李四喜欢在图书馆看书。", metadata={"source": "test.txt"}),
    ]
    
    # 存入数据库
    my_kb.add_documents(test_docs)
    print("文档已成功存入 ChromaDB")
    
    # 搜索
    query = "谁爱学习？"
    results = my_kb.search(query)
    
    # 打印搜索结果
    print(f"\n搜索问题: {query}")
    for i, doc in enumerate(results):
        print(f"匹配结果 {i+1}: {doc.page_content} (来源: {doc.metadata['source']})")
        
    # 删除集合，释放空间
    my_kb.delete_collection()
    print("\n测试完成，已删除集合。")
