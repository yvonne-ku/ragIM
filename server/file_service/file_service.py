import importlib

import os
from typing import Dict, Generator, List
from functools import lru_cache
from pathlib import Path
from langchain_core.documents import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, TextSplitter
from langchain_community.document_loaders import JSONLoader, TextLoader
import chardet

from server import settings
from server.file_service.text_splitter import (
    zh_title_enhance as func_zh_title_enhance,
)
from server.utils import StaticPathTools, logger


class StaticLoaderAndSplitterTools:
    """
    Loader 字典
    """
    LOADER_DICT = {
        "UnstructuredHTMLLoader": [".html", ".htm"],
        "MHTMLLoader": [".mhtml"],
        "TextLoader": [".md"],
        "UnstructuredMarkdownLoader": [".md"],
        "JSONLoader": [".json"],
        "JSONLinesLoader": [".jsonl"],
        "CSVLoader": [".csv"],
        # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv
        "RapidOCRPDFLoader": [".pdf"],
        "RapidOCRDocLoader": [".docx"],
        "RapidOCRPPTLoader": [
            ".ppt",
            ".pptx",
        ],
        "RapidOCRLoader": [".png", ".jpg", ".jpeg", ".bmp"],
        "UnstructuredFileLoader": [
            ".eml",
            ".msg",
            ".rst",
            ".rtf",
            ".txt",
            ".xml",
            ".epub",
            ".odt",
            ".tsv",
        ],
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
        "EverNoteLoader": [".enex"],
    }
    SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]

    @staticmethod
    def get_loaderClass(ext):
        for LoaderClass, extensions in StaticLoaderAndSplitterTools.LOADER_DICT.items():
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
                    "server.file_service.ocr_loader"
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
            
            # 3. 加载出错时使用 langchain 的 UnstructuredFileLoader
            document_loaders_module = importlib.import_module(
                "langchain_community.document_loaders"
            )
            DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

        # 为不同的 Loader 设置参数
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

        # 创建 Loader 实例
        loader = DocumentLoader(file_path, **loader_kwargs)
        return loader

    @staticmethod
    @lru_cache()
    def make_text_splitter(splitter_name, chunk_size, chunk_overlap):
        """
        根据指定参数动态创建文本切分器（TextSplitter）实例
        优先使用项目自定义切分器，无则使用langchain内置切分器，异常时兜底为通用递归字符切分器
        使用lru_cache缓存切分器实例，避免重复创建相同参数的切分器
        
        :param splitter_name: 切分器名称（如SpacyTextSplitter/RecursiveCharacterTextSplitter等）
        :param chunk_size: 每个文本块的最大长度（字符数/Token数，取决于切分器类型）
        :param chunk_overlap: 相邻文本块的重叠长度，保证上下文连贯性
        :return: 初始化完成的TextSplitter实例
        """
        # 切分器兜底
        if splitter_name == "" or splitter_name is None:
            splitter_name = "SpacyTextSplitter"
        try:
            # 1. 优先从自定义切分器模块获取
            try: 
                text_splitter_module = importlib.import_module("server.file_service.text_splitter")
                TextSplitter = getattr(text_splitter_module, splitter_name)

            # 2. 否则从 langchain 的切分器模块获取
            except:  
                text_splitter_module = importlib.import_module(
                    "langchain.text_splitter"
                )
                TextSplitter = getattr(text_splitter_module, splitter_name)

            # 3. 从配置中获取当前切分器的来源类型（tiktoken/huggingface/默认），来源不同参数不同
            # 3.1 来自 tiktoken 的切分器
            if (settings.kb_settings.text_splitter_dict[splitter_name]["source"] == "tiktoken"):
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=settings.kb_settings.text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=settings.kb_settings.text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
            # 3.2 来自 huggingface 的切分器
            elif (settings.kb_settings.text_splitter_dict[splitter_name]["source"] == "huggingface"):
                if (settings.kb_settings.text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "gpt2"):
                    from langchain.text_splitter import CharacterTextSplitter
                    from transformers import GPT2TokenizerFast
                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                else:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        settings.kb_settings.text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True,
                    )
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

            # 3.3 默认切分器
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
        except Exception as e:
            print(e)
            text_splitter_module = importlib.import_module("langchain.text_splitter")
            TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
            text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # If you use SpacyTextSplitter you can use GPU to do split likes Issue #1287
        # text_splitter._tokenizer.max_length = 37016792
        # text_splitter._tokenizer.prefer_gpu()
        return text_splitter

"""
一个 .txt 或者一个 .md 就是一个此类的实例
先通过 XXXLoader 将 .xxx 后缀的文件转为 doc 对象（self.docs）
再通过 splitter 将 doc 对象拆分成小片段（self.splitted_docs）
"""
class KnowledgeFile:
    ext: str = ""                   # 文件扩展名（小写）
    kb_name: str = ""               # 知识库名称
    doc_name: str = ""               # 标准化后的文件名
    doc_path: str = ""               # 文件完整路径
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
        # 文件名处理，并获取 ext
        self.kb_name = kb_name
        self.doc_name = str(Path(file_path).as_posix())    # 封装处理文件路径，兼容不同操作系统
        self.ext = os.path.splitext(file_path)[-1].lower()  
        if self.ext not in StaticLoaderAndSplitterTools.SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.doc_name}")
        self.doc_path = StaticPathTools.get_full_path(kb_name, self.doc_name)  # 完整的文件名
       
        # Loader 和 Spliter 按照 ext 设置
        self.loader_kwargs = loader_kwargs
        self.loader_name = StaticLoaderAndSplitterTools.get_loaderClass(self.ext)
        self.text_splitter_name = settings.kb_settings.TEXT_SPLITTER_NAME
        self.docs = None   
        self.splitted_docs = None

    def file2docs(self, refresh: bool = False):
        """ 根据文件类型，将文件内容 load 成 document。

        读取指定文件，解析内容并生成文档对象；
        当 refresh=True 时，会强制重新读取文件，忽略缓存。

        Args:
            refresh: 是否强制刷新/重新处理文件，默认值为 False（使用缓存）。

        Returns:
            list: 包含文档对象的列表，每个元素是一个文档实例。
        """
        if self.docs is None or refresh:
            logger.info(f"{self.loader_name} used for {self.doc_path}")
            # 拿到 Loader
            loader = StaticLoaderAndSplitterTools.get_loader(
                loader_name=self.loader_name,
                file_path=self.doc_path,
                loader_kwargs=self.loader_kwargs,
            )
            if isinstance(loader, TextLoader):
                loader.encoding = "utf8"
            # 使用 Loader 加载文件
            self.docs = loader.load()
        return self.docs

    def docs2texts(
        self,
        docs: List[Document] = None,
        zh_title_enhance: bool = settings.kb_settings.ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = settings.kb_settings.CHUNK_SIZE,
        chunk_overlap: int = settings.kb_settings.OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        """
        将文档对象切分为小文本块，支持中文标题增强，适用于 LLM 知识库构建
        :param docs: 待切分的 Document 文档列表，若为 None 则从文件加载
        :param zh_title_enhance: 是否开启中文标题增强（提升检索效果）
        :param refresh: 是否强制重新从文件加载文档（用于文件更新场景）
        :param chunk_size: 每个文本块的最大字符长度
        :param chunk_overlap: 相邻文本块的重叠字符长度（保证上下文连贯）
        :param text_splitter: 自定义文本切分器，未传则自动创建
        :return: 切分并优化后的 Document 文档列表
        """
        docs = docs or self.file2docs(refresh=refresh)
        # 检查文件是否为空，或者是否需要最实时的文档
        if not docs:
            return []
        
        # 仅对非 CSV 文件进行切分（CSV文件通常按行处理，无需额外切分）
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = StaticLoaderAndSplitterTools.make_text_splitter(
                    splitter_name=self.text_splitter_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            logger.info(
                f"{text_splitter.__class__.__name__} used for {self.doc_path} with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            # 检查切分结果是否为空

            # 通用切分器直接处理文档列表，返回切分后的小文档列表
            docs = text_splitter.split_documents(docs)

        # 检查切分结果是否为空
        if not docs:
            return []

        # 中文标题增强
        if zh_title_enhance:
            docs = func_zh_title_enhance(docs)

        self.splitted_docs = docs
        return self.splitted_docs


if __name__ == "__main__":

    # 创建文件处理对象
    kb_file = KnowledgeFile(
        file_path="d:\\MyProjects\\ragIM\\test\\samples\\1706.03762v7.pdf", 
        kb_name="samples"
    )

    # 转化为 doc
    docs = kb_file.file2docs()
    print(f"文档总数: {len(docs)}")

    # 切分 doc
    print("正在切分文档...")
    texts = kb_file.docs2texts(docs)
    print(f"切分完成，片段总数: {len(texts)}")
    if texts:
        for i, text in enumerate(texts[:3]):
            print(f"\n片段 {i + 1}:")
            print(f"元数据: {text.metadata}")
            print(f"内容: {text.page_content}")
        # 输出 txt 文件看看
        import os
        test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test")
        for i, text in enumerate(texts[:3]):
            output_file = os.path.join(test_dir, f"chunk_{i + 1}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"内容: {text.page_content}")
                f.write(f"\n\n元数据: {text.metadata}")
            print(f"已将片段 {i + 1} 写入到文件: {output_file}")
    