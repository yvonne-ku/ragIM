import importlib
import json
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlencode
from typing import Dict, Generator, List, Tuple, Union

import chardet
import langchain_community.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, TextSplitter
from langchain_community.document_loaders import JSONLoader, TextLoader

from chatchat.settings import Settings
from chatchat.server.file_rag.text_splitter import (
    zh_title_enhance as func_zh_title_enhance,
)
from chatchat.server.utils import run_in_process_pool, run_in_thread_pool



def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def list_kbs_from_folder():
    return [
        f
        for f in os.listdir(Settings.basic_settings.KB_ROOT_PATH)
        if os.path.isdir(os.path.join(Settings.basic_settings.KB_ROOT_PATH, f))
    ]


def list_files_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    result = []

    def is_skiped_path(path: str):
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False

    def process_entry(entry):
        if is_skiped_path(entry.path):
            return

        if entry.is_symlink():
            target_path = os.path.realpath(entry.path)
            with os.scandir(target_path) as target_it:
                for target_entry in target_it:
                    process_entry(target_entry)
        elif entry.is_file():
            file_path = Path(
                os.path.relpath(entry.path, doc_path)
            ).as_posix()  # 路径统一为 posix 格式
            result.append(file_path)
        elif entry.is_dir():
            with os.scandir(entry.path) as it:
                for sub_entry in it:
                    process_entry(sub_entry)

    with os.scandir(doc_path) as it:
        for entry in it:
            process_entry(entry)

    return result



# patch json.dumps to disable ensure_ascii
def _new_json_dumps(obj, **kwargs):
    kwargs["ensure_ascii"] = False
    return _origin_json_dumps(obj, **kwargs)


if json.dumps is not _new_json_dumps:
    _origin_json_dumps = json.dumps
    json.dumps = _new_json_dumps


class JSONLinesLoader(JSONLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_lines = True


langchain_community.document_loaders.JSONLinesLoader = JSONLinesLoader



@lru_cache()
def make_text_splitter(splitter_name, chunk_size, chunk_overlap):
    """
    根据参数获取特定的分词器
    """
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if (
            splitter_name == "MarkdownHeaderTextSplitter"
        ):  # MarkdownHeaderTextSplitter特殊判定
            headers_to_split_on = Settings.kb_settings.text_splitter_dict[splitter_name][
                "headers_to_split_on"
            ]
            text_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on, strip_headers=False
            )
        else:
            try:  # 优先使用用户自定义的text_splitter
                text_splitter_module = importlib.import_module("chatchat.server.file_rag.text_splitter")
                TextSplitter = getattr(text_splitter_module, splitter_name)
            except:  # 否则使用langchain的text_splitter
                text_splitter_module = importlib.import_module(
                    "langchain.text_splitter"
                )
                TextSplitter = getattr(text_splitter_module, splitter_name)

            if (
                Settings.kb_settings.text_splitter_dict[splitter_name]["source"] == "tiktoken"
            ):  # 从tiktoken加载
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=Settings.kb_settings.text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=Settings.kb_settings.text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
            elif (
                Settings.kb_settings.text_splitter_dict[splitter_name]["source"] == "huggingface"
            ):  # 从huggingface加载
                if (
                    Settings.kb_settings.text_splitter_dict[splitter_name]["tokenizer_name_or_path"]
                    == "gpt2"
                ):
                    from langchain.text_splitter import CharacterTextSplitter
                    from transformers import GPT2TokenizerFast

                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                else:  # 字符长度加载
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        Settings.kb_settings.text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True,
                    )
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
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


class KnowledgeFile:

    def docs2texts(
        self,
        docs: List[Document] = None,
        zh_title_enhance: bool = Settings.kb_settings.ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        docs = docs or self.file2docs(refresh=refresh)
        if not docs:
            return []
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = make_text_splitter(
                    splitter_name=self.text_splitter_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                docs = text_splitter.split_documents(docs)

        if not docs:
            return []

        print(f"文档切分示例：{docs[0]}")
        if zh_title_enhance:
            docs = func_zh_title_enhance(docs)
        self.splited_docs = docs
        return self.splited_docs

    def file2text(
        self,
        zh_title_enhance: bool = Settings.kb_settings.ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
        chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(
                docs=docs,
                zh_title_enhance=zh_title_enhance,
                refresh=refresh,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
        return self.splited_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)


def files2docs_in_thread_file2docs(
    *, file: KnowledgeFile, **kwargs
) -> Tuple[bool, Tuple[str, str, List[Document]]]:
    try:
        return True, (file.kb_name, file.filename, file.file2text(**kwargs))
    except Exception as e:
        msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
        logger.error(f"{e.__class__.__name__}: {msg}")
        return False, (file.kb_name, file.filename, msg)


def files2docs_in_thread(
    files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
    chunk_size: int = Settings.kb_settings.CHUNK_SIZE,
    chunk_overlap: int = Settings.kb_settings.OVERLAP_SIZE,
    zh_title_enhance: bool = Settings.kb_settings.ZH_TITLE_ENHANCE,
) -> Generator:
    """
    利用多线程批量将磁盘文件转化成langchain Document.
    如果传入参数是Tuple，形式为(filename, kb_name)
    生成器返回值为 status, (kb_name, file_name, docs | error)
    """

    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            if isinstance(file, tuple) and len(file) >= 2:
                filename = file[0]
                kb_name = file[1]
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                kwargs.update(file)
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            kwargs["file"] = file
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
            kwargs["zh_title_enhance"] = zh_title_enhance
            kwargs_list.append(kwargs)
        except Exception as e:
            yield False, (kb_name, filename, str(e))

    for result in run_in_thread_pool(
        func=files2docs_in_thread_file2docs, params=kwargs_list
    ):
        yield result


def format_reference(kb_name: str, docs: List[Dict], api_base_url: str="") -> List[Dict]:
    '''
    将知识库检索结果格式化为参考文档的格式
    '''
    from chatchat.server.utils import api_address
    api_base_url = api_base_url or api_address(is_public=True)

    source_documents = []
    for inum, doc in enumerate(docs):
        filename = doc.get("metadata", {}).get("source")
        parameters = urlencode(
            {
                "knowledge_base_name": kb_name,
                "file_name": filename,
            }
        )
        api_base_url = api_base_url.strip(" /")
        url = (
            f"{api_base_url}/knowledge_base/download_doc?" + parameters
        )
        page_content = doc.get("page_content")
        ref = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{page_content}\n\n"""
        source_documents.append(ref)
    
    return source_documents
