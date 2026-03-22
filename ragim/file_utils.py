# ProactiveDialogDataset

import importlib
from typing import Dict, Generator, List, Tuple, Union
import os
from pathlib import Path
from langchain.text_splitter import MarkdownHeaderTextSplitter, TextSplitter
from langchain_community.document_loaders import JSONLoader, TextLoader
from pprint import pprint
from utils import logger
from ragim.settings import Settings


class StaticPathTools:
    @staticmethod
    def get_kb_path(kb_name: str):
        return os.path.join(Settings.basic_settings.KB_ROOT_PATH, kb_name)
        
    @staticmethod
    def get_doc_path(kb_name: str):
        return os.path.join(StaticPathTools.get_kb_path(kb_name), "content")

    @staticmethod
    def get_vec_path(kb_name: str, vector_name: str):
        return os.path.join(StaticPathTools.get_kb_path(kb_name), "vector_store", vector_name)
    
    @staticmethod
    def get_full_path(kb_name: str, doc_name: str):
        doc_path = Path(StaticPathTools.get_doc_path(kb_name)).resolve()    # 转化为绝对路径
        full_path = (doc_path / doc_name).resolve()
        if str(full_path).startswith(str(doc_path)):
            return str(full_path)


class StaticLoaderTools:
    """
    Loader 字典
    """
    LOADER_DICT = {
        "TextLoader": [".md"],
        "JSONLoader": [".json"],
        "JSONLinesLoader": [".jsonl"],
        "CSVLoader": [".csv"],

        "RapidOCRPDFLoader": [".pdf"],
        "RapidOCRDocLoader": [".docx"],
        "RapidOCRPPTLoader": [".ppt", ".pptx"],
        "RapidOCRLoader": [".png", ".jpg", ".jpeg", ".bmp"],

        "UnstructuredHTMLLoader": [".html", ".htm"],
        "UnstructuredMarkdownLoader": [".md"],
        "UnstructuredEmailLoader": [".eml", ".msg"],
        "UnstructuredEPubLoader": [".epub"],
        "UnstructuredExcelLoader": [".xlsx", ".xls", ".xlsd"],
        "NotebookLoader": [".ipynb"],
        "UnstructuredODTLoader": [".odt"],
        "PythonLoader": [".py"],
        "UnstructuredRSTLoader": [".rst"],
        "UnstructuredRTFLoader": [".rtf"],
        "SRTLoader": [".srt"],
        "TomlLoader": [".toml"],
        "UnstructuredTSVLoader": [".tsv"],
        "UnstructuredWordDocumentLoader": [".docx"],
        "UnstructuredXMLLoader": [".xml"],
        "UnstructuredPowerPointLoader": [".ppt", ".pptx"],
    }
    SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]

    @staticmethod
    def get_loaderClass(ext):
        for LoaderClass, extensions in StaticLoaderTools.LOADER_DICT.items():
            if ext in extensions:
                return LoaderClass
    
    @staticmethod
    def get_loader(loader_name: str, file_path: str, loader_kwargs: Dict = None):
        loader_kwargs = loader_kwargs or {}
        try:
            if loader_name in [
                "RapidOCRPDFLoader",
                "RapidOCRLoader",
                "RapidOCRDocLoader",
                "RapidOCRPPTLoader",
            ]:
                # 1. 如果是 OCR，使用 ragim.ocr_loader
                document_loaders_module = importlib.import_module(
                    "ragim.ocr_loader"
                )
            else:
                # 2. 如果是普通文件 Loader，直接使用 langchain
                document_loaders_module = importlib.import_module(
                    "langchain_community.document_loaders"
                )
            DocumentLoader = getattr(document_loaders_module, loader_name)
        except Exception as e:
            msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
            logger.error(f"{e.__class__.__name__}: {msg}")
            document_loaders_module = importlib.import_module(
                "langchain_community.document_loaders"
            )
            DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

        if loader_name == "UnstructuredFileLoader":
            loader_kwargs.setdefault("autodetect_encoding", True)
        elif loader_name == "CSVLoader":
            if not loader_kwargs.get("encoding"):
                # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
                with open(file_path, "rb") as struct_file:
                    encode_detect = chardet.detect(struct_file.read())
                if encode_detect is None:
                    encode_detect = {"encoding": "utf-8"}
                loader_kwargs["encoding"] = encode_detect["encoding"]

        elif loader_name == "JSONLoader":
            loader_kwargs.setdefault("jq_schema", ".")
            loader_kwargs.setdefault("text_content", False)
        elif loader_name == "JSONLinesLoader":
            loader_kwargs.setdefault("jq_schema", ".")
            loader_kwargs.setdefault("text_content", False)

        loader = DocumentLoader(file_path, **loader_kwargs)
        return loader


"""
先通过 XXXLoader 将 .xxx 后缀的文件转为 doc 对象（self.docs）
再通过 splitter 将 doc 对象拆分成小片段（self.splited_docs）
"""
class KnowledgeFile:
    ext: str = ""                   # 文件扩展名（小写）
    kb_name: str = ""               # 知识库名称
    docname: str = ""              # 标准化后的文件名
    docpath: str = ""               # 文件完整路径
    loader_kwargs: Dict = {}        # 加载器配置参数
    loader_name: str = ""           # 文档加载器名称
    text_splitter_name: str = ""    # 文本分割器名称


    def __init__(
            self,
            file_path: str,
            kb_name: str,
            loader_kwargs: Dict = {},
    ):
        """ 初始化 KnowledgeFile 实例。

        Args:
            file_url: 文件的 URL 或路径。
            knowledge_base_name: 关联的知识库名称。
            loader_kwargs: 加载器的参数字典，默认为空。
        """
        # 文件名处理
        self.kb_name = kb_name
        self.docname = str(Path(file_path).as_posix())    # 封装处理文件路径，兼容不同操作系统
        self.ext = os.path.splitext(file_path)[-1].lower()  
        if self.ext not in StaticLoaderTools.SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.docname}")
        self.docpath = StaticPathTools.get_full_path(kb_name, self.docname)  # 完整的文件名
       
        # Loader 和 Spliter 设置
        self.loader_kwargs = loader_kwargs
        self.doc_loader_name = StaticLoaderTools.get_LoaderClass(self.ext)
        self.text_splitter_name = Settings.kb_settings.TEXT_SPLITTER_NAME
        self.docs = None   
        self.splited_docs = None


    def file2docs(self, refresh: bool = False):
        """ 将文件内容转换为文档对象列表。

        读取指定文件，解析内容并生成文档对象；
        当 refresh=True 时，会强制重新读取文件，忽略缓存。

        Args:
            refresh: 是否强制刷新/重新处理文件，默认值为 False（使用缓存）。

        Returns:
            list: 包含文档对象的列表，每个元素是一个文档实例。
        """
        if self.docs is None or refresh:
            logger.info(f"{self.loader_name} used for {self.docpath}")
            # 拿到 Loader
            loader = StaticLoaderTools.get_loader(
                loader_name=self.loader_name,
                file_path=self.docpath,
                loader_kwargs=self.loader_kwargs,
            )
            if isinstance(loader, TextLoader):
                loader.encoding = "utf8"
            self.docs = loader.load()
        return self.docs




if __name__ == "__main__":

    # 创建文件处理对象
    kb_file = KnowledgeFile(
        file_path="D:\\MyProjects\\ragIM\\data\\raw_data\\ProactiveDialogDataset\\data_txt\\scene_6f13d1\\num_task_1\\dialogue_0.txt",
        kb_file="test")
    
    # 转化为 doc
    docs = kb_file.file2docs()
    
    # 切分 doc
    texts = kb_file.docs2splited(docs)

    for text in texts:
        print(text)

