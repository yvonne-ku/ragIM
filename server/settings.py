from __future__ import annotations

import os
import sys
from pathlib import Path
import typing as t

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

from server import __version__

# 计算项目根目录，基于当前文件的位置
SERVER_ROOT: Path = Path(__file__).parent.resolve()
RAGIM_ROOT: Path = SERVER_ROOT.parent.resolve()


# 加载 .env 环境变量
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


# 项目基础配置
class BasicSettings(BaseSettings):

    model_config = SettingsConfigDict(yaml_file=SERVER_ROOT / "basic_settings.yaml")
    """会根据该配置文件合并默认值生成项目代码的版本"""

    # 项目目录结构信息
    DATA_ROOT: Path = RAGIM_ROOT / "data"
    OUTPUT_PATH: Path = DATA_ROOT / "output_results"
    VS_PATH: Path = DATA_ROOT / "vector_store"
    RAW_JSON_PATH: Path = DATA_ROOT / "raw_json_data"
    CHUNKS_PATH: Path = DATA_ROOT / "processed_chunks"
    NLTK_DATA_PATH: Path = RAGIM_ROOT / "resources" / "nltk_data"    # NLTK 数据目录
    os.environ["NLTK_DATA"] = str(NLTK_DATA_PATH)

    def make_dirs(self):
        """初始化项目所有需要的目录"""
        for path in [self.DATA_ROOT, self.OUTPUT_PATH, self.VS_PATH, self.RAW_JSON_PATH, self.CHUNKS_PATH, self.NLTK_DATA_PATH]:
            if not os.path.exists(path):
                os.makedirs(path)

    # 服务器信息
    OPEN_CROSS_DOMAIN: bool = False
    """API 是否开启跨域"""

    DEFAULT_BIND_HOST: str = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"
    """
    允许访问的主机地址
    0.0.0.0 表示监听所有网卡上的请求，所以访问请求可以打在本机的所有网卡上
    127.0.0.1 表示仅监听从本机回环地址来的请求，也就是仅本机开发访问使用
    这里的判断逻辑是：如果是 windows 系统，就监听回环地址，否则监听所有网卡
    """

    API_SERVER: dict = {"host": DEFAULT_BIND_HOST, "port": 7861, "public_host": "127.0.0.1", "public_port": 7861}
    """API 服务器地址"""

    WEBUI_SERVER: dict = {"host": DEFAULT_BIND_HOST, "port": 8501}
    """WEB UI 服务器地址"""



# 知识库相关配置
class KBSettings(BaseSettings):

    model_config = SettingsConfigDict(yaml_file=SERVER_ROOT / "kb_settings.yaml")

    DEFAULT_KNOWLEDGE_BASE: str = "samples"
    """默认使用的知识库"""

    DEFAULT_VS_TYPE: t.Literal["faiss", "milvus", "zilliz", "pg", "es", "relyt", "chromadb"] = "chromadb"
    """默认向量库/全文检索引擎类型"""

    CHUNK_SIZE: int = 750
    """知识库中单段文本长度(不适用MarkdownHeaderTextSplitter)"""

    OVERLAP_SIZE: int = 150
    """
    知识库中相邻文本重合长度(不适用MarkdownHeaderTextSplitter)
    比如第一段是 1-750 字，第二段是 600-1350 字（750-150=600），避免分割后语义断裂
    """

    VECTOR_SEARCH_TOP_K: int = 3
    """知识库匹配向量数量"""

    SCORE_THRESHOLD: float = 0.5
    """知识库匹配相关度阈值，取值范围在0-2之间，SCORE越小，相关度越高，取到2相当于不筛选，建议设置在0.5左右"""

    ZH_TITLE_ENHANCE: bool = False
    """是否开启中文标题加强，以及标题增强的相关配置"""

    PDF_OCR_THRESHOLD: t.Tuple[float, float] = (0.6, 0.6)
    """
    PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
    这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
    """

    kbs_config: t.Dict[str, t.Dict] = {
        "faiss": {},
        "milvus": {
            "host": "127.0.0.1",
            "port": "19530",
            "user": "",
            "password": "",
            "secure": False
        },
        "zilliz": {
            "host": "in01-a7ce524e41e3935.ali-cn-hangzhou.vectordb.zilliz.com.cn",
            "port": "19530",
            "user": "",
            "password": "",
            "secure": True
        },
        "pg": {
            "connection_uri": "postgresql://postgres:postgres@127.0.0.1:5432/langchain_chatchat"
        },
        "relyt": {
            "connection_uri": "postgresql+psycopg2://postgres:postgres@127.0.0.1:7000/langchain_chatchat"
        },
        "es": {
            "scheme": "http",
            "host": "127.0.0.1",
            "port": "9200",
            "index_name": "test_index",
            "user": "",
            "password": "",
            "verify_certs": True,
            "ca_certs": None,
            "client_cert": None,
            "client_key": None
        },
        "milvus_kwargs": {
            "search_params": {
                "metric_type": "L2"
            },
            "index_params": {
                "metric_type": "L2",
                "index_type": "HNSW"
            }
        },
        "chromadb": {}
    }
    """
    向量库配置
    针对不同的向量库有不同的配置
    """

    text_splitter_dict: t.Dict[str, t.Dict[str, t.Any]] = {
        "ChineseRecursiveTextSplitter": {
            "source": "",
            "tokenizer_name_or_path": "",
        },
        "SpacyTextSplitter": {
            "source": "huggingface",
            "tokenizer_name_or_path": "gpt2",
        },
        "RecursiveCharacterTextSplitter": {
            "source": "tiktoken",
            "tokenizer_name_or_path": "cl100k_base",
        },
        "MarkdownHeaderTextSplitter": {
            "headers_to_split_on": [
                ("#", "head1"),
                ("##", "head2"),
                ("###", "head3"),
                ("####", "head4"),
            ]
        },
    }
    """
    为不同的文本分割器提供配置参数
    """

    TEXT_SPLITTER_NAME: str = "RecursiveCharacterTextSplitter"
    """指定默认使用的文本分割器，是英文默认分割器"""

    EMBEDDING_KEYWORD_FILE: str = "embedding_keywords.txt"
    """指定 Embedding 模型（向量模型）的自定义词表文件路径 —— 可以添加领域专属词汇，提升模型对专业术语的向量表示精度。"""


# 定义模型加载平台相关配置
class PlatformConfig(BaseModel):
    """模型加载平台配置"""

    platform_name: str = "zhipuai"
    """平台名称"""

    platform_type: t.Literal["xinference", "ollama", "oneapi", "fastchat", "openai", "custom openai"] = "custom openai"
    """平台类型"""

    api_llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    """openai api url"""

    api_embedding_base_url: str = "https://open.bigmodel.cn/api/paas/v4/embeddings"
    """openai api url"""

    api_key: str = os.getenv("ZHIPUAI_API_KEY")
    """api key if available"""


# 定义模型配置项
class ApiModelSettings(BaseSettings):
    """模型配置项"""

    model_config = SettingsConfigDict(yaml_file=RAGIM_ROOT / "model_settings.yaml")

    DEFAULT_LLM_MODEL: str = "glm-4-plus"
    """默认选用的 LLM 名称"""

    DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    """默认选用的 Embedding 名称"""

    TEMPERATURE: float = 0.7
    """LLM 通用对话参数"""

    LLM_MODEL_CONFIG: t.Dict[str, t.Dict] = {
        # 意图识别不需要输出，模型后台知道就行
        "preprocess_model": {
            "model": "",
            "temperature": 0.05,
            "max_tokens": 4096,
            "history_len": 10,
            "prompt_name": "default",
            "callbacks": False,
        },
        "llm_model": {
            "model": "",
            "temperature": 0.9,
            "max_tokens": 4096,
            "history_len": 10,
            "prompt_name": "default",
            "callbacks": True,
        },
        "action_model": {
            "model": "",
            "temperature": 0.01,
            "max_tokens": 4096,
            "history_len": 10,
            "prompt_name": "ChatGLM3",
            "callbacks": True,
        },
        "postprocess_model": {
            "model": "",
            "temperature": 0.01,
            "max_tokens": 4096,
            "history_len": 10,
            "prompt_name": "default",
            "callbacks": True,
        },
        "image_model": {
            "model": "sd-turbo",
            "size": "256*256",
        },
    }
    """
    LLM 模型配置，
    包括了不同用途的 LLM 初始化参数。
    """

    MODEL_PLATFORMS: t.List[PlatformConfig] = [
        PlatformConfig(**{
            "platform_name": "zhipuai",
            "api_embedding_base_url": "https://open.bigmodel.cn/api/paas/v4",
            "api_key": os.getenv("ZHIPUAI_API_KEY", ""),
        }),
        PlatformConfig(**{
            "platform_name": "openai",
            "api_embedding_base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
        }),
    ]
    """模型平台配置"""


# 定义 Prompt 模板
class PromptSettings(BaseSettings):

    model_config = SettingsConfigDict(yaml_file=RAGIM_ROOT / "prompt_settings.yaml",
                                      json_file=RAGIM_ROOT / "prompt_settings.json",
                                      extra="allow")

    rag: dict = {
        "default": (
            "【指令】根据已知信息，简洁和专业的来回答问题。"
            "如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。\n\n"
            "【已知信息】{{context}}\n\n"
            "【问题】{{question}}\n"
        ),
        "empty": (
            "请你回答我的问题:\n"
            "{{question}}"
        ),
    }
    '''RAG 用模板，可用于知识库问答、文件对话、搜索引擎对话'''



basic_settings = BasicSettings()
kb_settings = KBSettings()
platform_config = PlatformConfig()
api_model_settings = ApiModelSettings()
prompt_settings = PromptSettings()