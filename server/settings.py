from __future__ import annotations

import os
import sys
from pathlib import Path
import typing as t

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from server import __version__

RAGIM_ROOT: Path = Path(os.environ.get("RAGIM_ROOT", "..")).resolve()
SERVER_ROOT: Path = RAGIM_ROOT / "server"


# 项目基础配置
class BasicSettings(BaseSettings):

    model_config = SettingsConfigDict(yaml_file=SERVER_ROOT / "basic_settings.yaml")
    """会根据该配置文件合并默认值生成项目代码的版本"""

    # 项目目录结构信息
    DATA_ROOT: Path = RAGIM_ROOT / "data"
    NLTK_DATA_PATH: Path = DATA_ROOT / "nltk_data"    # NLTK 数据目录
    KB_ROOT: Path = DATA_ROOT / "knowledge_base"      # 知识库根目录
    MEDIA_PATH: Path = DATA_ROOT / "media"            # 模型生成内容（图片、视频、音频等）保存位置
    LOG_ROOT: Path = DATA_ROOT / "logs"
    TEMP_PATH: Path = DATA_ROOT / "temp"               # 临时文件目录

    os.environ["NLTK_DATA"] = str(NLTK_DATA_PATH)

    def make_dirs(self):
        '''创建所有数据目录'''
        for p in [
            self.DATA_ROOT,
            self.MEDIA_PATH,
            self.LOG_ROOT,
            self.TEMP_PATH,
        ]:
            p.mkdir(parents=True, exist_ok=True)
        for n in ["image", "audio", "video"]:
            (self.MEDIA_PATH / n).mkdir(parents=True, exist_ok=True)
        Path(self.KB_ROOT).mkdir(parents=True, exist_ok=True)


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

    SQLALCHEMY_DATABASE_URI: str = "sqlite:///" + str(KB_ROOT / "info.db")
    """内嵌DB，记录KB的元数据"""


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

    # TEXT_SPLITTER_NAME: str = "ChineseRecursiveTextSplitter"
    TEXT_SPLITTER_NAME: str = "RecursiveCharacterTextSplitter"
    """指定默认使用的文本分割器，第一个是中文默认分割器，第二个是英文默认分割器"""

    EMBEDDING_KEYWORD_FILE: str = "embedding_keywords.txt"
    """指定 Embedding 模型（向量模型）的自定义词表文件路径 —— 可以添加领域专属词汇，提升模型对专业术语的向量表示精度。"""


# 定义模型加载平台相关配置
class PlatformConfig(BaseModel):
    """模型加载平台配置"""

    platform_name: str = "zhipuai"
    """平台名称"""

    platform_type: t.Literal["xinference", "ollama", "oneapi", "fastchat", "openai", "custom openai"] = "custom openai"
    """平台类型"""

    api_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    """openai api url"""

    api_key: str = os.getenv("ZHIPUAI_API_KEY")
    """api key if available"""


# 定义模型配置项
class ApiModelSettings(BaseSettings):
    """模型配置项"""

    model_config = SettingsConfigDict(yaml_file=RAGIM_ROOT / "model_settings.yaml")

    DEFAULT_LLM_MODEL: str = "glm-4-plus"
    """默认选用的 LLM 名称"""

    DEFAULT_EMBEDDING_MODEL: str = "embedding-3"
    """默认选用的 Embedding 名称"""

    Agent_MODEL: str = ""
    """AgentLM模型的名称 (可以不指定，指定之后就锁定进入Agent之后的Chain的模型，不指定就是 DEFAULT_LLM_MODEL)"""

    HISTORY_LEN: int = 3
    """默认历史对话轮数"""

    MAX_TOKENS: t.Optional[int] = None
    """大模型最长支持的长度，如果不填写，则使用模型默认的最大长度，如果填写，则为用户设定的最大长度"""

    TEMPERATURE: float = 0.7
    """LLM通用对话参数"""

    SUPPORT_AGENT_MODELS: t.List[str] = [
        "chatglm3-6b",
        "glm-4",
        "openai-api",
        "Qwen-2",
        "qwen2-instruct",
        "gpt-3.5-turbo",
        "gpt-4o",
    ]
    """支持的Agent模型"""

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
    LLM模型配置，包括了不同模态初始化参数。
    `model` 如果留空则自动使用 DEFAULT_LLM_MODEL
    """

    MODEL_PLATFORMS: t.List[PlatformConfig] = [
        PlatformConfig(**{
            "platform_name": "xinference",
            "platform_type": "xinference",
            "api_base_url": "http://127.0.0.1:9997/v1",
            "api_key": "EMPTY",
            "api_concurrencies": 5,
            "auto_detect_model": True,
            "llm_models": [],
            "embed_models": [],
            "text2image_models": [],
            "image2text_models": [],
            "rerank_models": [],
            "speech2text_models": [],
            "text2speech_models": [],
        }),
        PlatformConfig(**{
            "platform_name": "ollama",
            "platform_type": "ollama",
            "api_base_url": "http://127.0.0.1:11434/v1",
            "api_key": "EMPTY",
            "api_concurrencies": 5,
            "llm_models": [
                "qwen:7b",
                "qwen2:7b",
            ],
            "embed_models": [
                "quentinz/bge-large-zh-v1.5",
            ],
        }),
        PlatformConfig(**{
            "platform_name": "oneapi",
            "platform_type": "oneapi",
            "api_base_url": "http://127.0.0.1:3000/v1",
            "api_key": "sk-",
            "api_concurrencies": 5,
            "llm_models": [
                # 智谱 API
                "chatglm_pro",
                "chatglm_turbo",
                "chatglm_std",
                "chatglm_lite",
                # 千问 API
                "qwen-turbo",
                "qwen-plus",
                "qwen-max",
                "qwen-max-longcontext",
                # 千帆 API
                "ERNIE-Bot",
                "ERNIE-Bot-turbo",
                "ERNIE-Bot-4",
                # 星火 API
                "SparkDesk",
            ],
            "embed_models": [
                # 千问 API
                "text-embedding-v1",
                # 千帆 API
                "Embedding-V1",
            ],
            "text2image_models": [],
            "image2text_models": [],
            "rerank_models": [],
            "speech2text_models": [],
            "text2speech_models": [],
        }),
        PlatformConfig(**{
            "platform_name": "openai",
            "platform_type": "openai",
            "api_base_url": "https://api.openai.com/v1",
            "api_key": "sk-proj-",
            "api_concurrencies": 5,
            "llm_models": [
                "gpt-4o",
                "gpt-3.5-turbo",
            ],
            "embed_models": [
                "text-embedding-3-small",
                "text-embedding-3-large",
            ],
        }),
    ]
    """模型平台配置"""


# 定义 Agent 工具配置项
class ToolSettings(BaseSettings):
    """Agent 工具配置项"""
    model_config = SettingsConfigDict(yaml_file=RAGIM_ROOT / "tool_settings.yaml",
                                      json_file=RAGIM_ROOT / "tool_settings.json",
                                      extra="allow")

    search_local_knowledgebase: dict = {
        "use": False,
        "top_k": 3,
        "score_threshold": 2.0,
        "conclude_prompt": {
            "with_result": '<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题"，'
                           "不允许在答案中添加编造成分，答案请使用中文。 </指令>\n"
                           "<已知信息>{{ context }}</已知信息>\n"
                           "<问题>{{ question }}</问题>\n",
            "without_result": "请你根据我的提问回答我的问题:\n"
                              "{{ question }}\n"
                              "请注意，你必须在回答结束后强调，你的回答是根据你的经验回答而不是参考资料回答的。\n",
        },
    }
    '''本地知识库工具配置项'''

    search_internet: dict = {
        "use": False,
        "search_engine_name": "duckduckgo",
        "search_engine_config": {
            "bing": {
                "bing_search_url": "https://api.bing.microsoft.com/v7.0/search",
                "bing_key": "",
            },
            "metaphor": {
                "metaphor_api_key": "",
                "split_result": False,
                "chunk_size": 500,
                "chunk_overlap": 0,
            },
            "duckduckgo": {},
            "searx": {
                "host": "https://metasearx.com",
                "engines": [],
                "categories": [],
                "language": "zh-CN",
            }
        },
        "top_k": 5,
        "verbose": "Origin",
        "conclude_prompt": "<指令>这是搜索到的互联网信息，请你根据这些信息进行提取并有调理，简洁的回答问题。如果无法从中得到答案，请说 “无法搜索到能回答问题的内容”。 "
                           "</指令>\n<已知信息>{{ context }}</已知信息>\n"
                           "<问题>\n"
                           "{{ question }}\n"
                           "</问题>\n",
    }
    '''搜索引擎工具配置项。推荐自己部署 searx 搜索引擎，国内使用最方便。'''

    arxiv: dict = {
        "use": False,
    }

    weather_check: dict = {
        "use": False,
        "api_key": "",
    }
    '''心知天气（https://www.seniverse.com/）工具配置项'''

    search_youtube: dict = {
        "use": False,
    }

    wolfram: dict = {
        "use": False,
        "appid": "",
    }

    calculate: dict = {
        "use": False,
    }
    '''numexpr 数学计算工具配置项'''

    text2images: dict = {
        "use": False,
        "model": "sd-turbo",
        "size": "256*256",
    }
    '''图片生成工具配置项。model 必须是在 model_settings.yaml/MODEL_PLATFORMS 中配置过的。'''

    text2sql: dict = {
        # 该工具需单独指定使用的大模型，与用户前端选择使用的模型无关
        "model_name": "qwen-plus",
        "use": False,
        # SQLAlchemy连接字符串，支持的数据库有：
        # crate、duckdb、googlesql、mssql、mysql、mariadb、oracle、postgresql、sqlite、clickhouse、prestodb
        # 不同的数据库请查阅SQLAlchemy用法，修改sqlalchemy_connect_str，配置对应的数据库连接，如sqlite为sqlite:///数据库文件路径，下面示例为mysql
        # 如提示缺少对应数据库的驱动，请自行通过poetry安装
        "sqlalchemy_connect_str": "mysql+pymysql://用户名:密码@主机地址/数据库名称",
        # 务必评估是否需要开启read_only,开启后会对sql语句进行检查，请确认text2sql.py中的intercept_sql拦截器是否满足你使用的数据库只读要求
        # 优先推荐从数据库层面对用户权限进行限制
        "read_only": False,
        # 限定返回的行数
        "top_k": 50,
        # 是否返回中间步骤
        "return_intermediate_steps": True,
        # 如果想指定特定表，请填写表名称，如["sys_user","sys_dept"]，不填写走智能判断应该使用哪些表
        "table_names": [],
        # 对表名进行额外说明，辅助大模型更好的判断应该使用哪些表，尤其是SQLDatabaseSequentialChain模式下,是根据表名做的预测，很容易误判。
        "table_comments": {
            # 如果出现大模型选错表的情况，可尝试根据实际情况填写表名和说明
            # "tableA":"这是一个用户表，存储了用户的基本信息",
            # "tableB":"角色表",
        },
    }
    '''
    text2sql使用建议
    1、因大模型生成的sql可能与预期有偏差，请务必在测试环境中进行充分测试、评估；
    2、生产环境中，对于查询操作，由于不确定查询效率，推荐数据库采用主从数据库架构，让text2sql连接从数据库，防止可能的慢查询影响主业务；
    3、对于写操作应保持谨慎，如不需要写操作，设置read_only为True,最好再从数据库层面收回数据库用户的写权限，防止用户通过自然语言对数据库进行修改操作；
    4、text2sql与大模型在意图理解、sql转换等方面的能力有关，可切换不同大模型进行测试；
    5、数据库表名、字段名应与其实际作用保持一致、容易理解，且应对数据库表名、字段进行详细的备注说明，帮助大模型更好理解数据库结构；
    6、若现有数据库表名难于让大模型理解，可配置下面table_comments字段，补充说明某些表的作用。
    '''

    amap: dict = {
        "use": False,
        "api_key": "高德地图 API KEY",
    }
    '''高德地图、天气相关工具配置项。'''

    text2promql: dict = {
        "use": False,
        # <your_prometheus_ip>:<your_prometheus_port>
        "prometheus_endpoint": "http://127.0.0.1:9090",
        # <your_prometheus_username>
        "username": "",
        # <your_prometheus_password>
        "password": "",
    }
    '''
    text2promql 使用建议
    1、因大模型生成的 promql 可能与预期有偏差, 请务必在测试环境中进行充分测试、评估;
    2、text2promql 与大模型在意图理解、metric 选择、promql 转换等方面的能力有关, 可切换不同大模型进行测试;
    3、当前仅支持 单prometheus 查询, 后续考虑支持 多prometheus 查询.
    '''

    url_reader: dict = {
        "use": False,
        "timeout": "10000",
    }
    '''URL内容阅读（https://r.jina.ai/）工具配置项
    请确保部署的网络环境良好，以免造成超时等问题'''


# 定义 Prompt 模板
class PromptSettings(BaseSettings):
    """Prompt 模板.除 Agent 模板使用 f-string 外，其它均使用 jinja2 格式"""

    model_config = SettingsConfigDict(yaml_file=RAGIM_ROOT / "prompt_settings.yaml",
                                      json_file=RAGIM_ROOT / "prompt_settings.json",
                                      extra="allow")

    preprocess_model: dict = {
        "default": (
            "你只要回复0 和 1 ，代表不需要使用工具。以下几种问题不需要使用工具:\n"
            "1. 需要联网查询的内容\n"
            "2. 需要计算的内容\n"
            "3. 需要查询实时性的内容\n"
            "如果我的输入满足这几种情况，返回1。其他输入，请你回复0，你只要返回一个数字\n"
            "这是我的问题:"
        ),
    }
    """意图识别用模板"""

    llm_model: dict = {
        "default": "{{input}}",
        "with_history": (
            "The following is a friendly conversation between a human and an AI.\n"
            "The AI is talkative and provides lots of specific details from its context.\n"
            "If the AI does not know the answer to a question, it truthfully says it does not know.\n\n"
            "Current conversation:\n"
            "{{history}}\n"
            "Human: {{input}}\n"
            "AI:"
        ),
    }
    '''普通 LLM 用模板'''

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

    action_model: dict = {
        "default": {
            "SYSTEM_PROMPT": (
                "You are a helpful assistant"
            ),
        },
        "openai-functions": {
            "SYSTEM_PROMPT": (
                "You are a helpful assistant"
            ),
            "HUMAN_MESSAGE": (
                "{input}"
            )
        },
        "glm3": {
            "SYSTEM_PROMPT": ("\nAnswer the following questions as best as you can. You have access to the following "
                              "tools:\n{tools}"),
            "HUMAN_MESSAGE": "Let's start! Human:{input}\n\n{agent_scratchpad}"

        },
        "qwen": {
            "SYSTEM_PROMPT": (
                "Answer the following questions as best you can. You have access to the following APIs:\n\n"
                "{tools}\n\n"
                "Use the following format:\n\n"
                "Question: the input question you must answer\n"
                "Thought: you should always think about what to do\n"
                "Action: the action to take, should be one of [{tool_names}]\n"
                "Action Input: the input to the action\n"
                "Observation: the result of the action\n"
                "... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\n"
                "Thought: I now know the final answer\n"
                "Final Answer: the final answer to the original input question\n\n"
                "Format the Action Input as a JSON object.\n\n"
                "Begin!\n\n"),
            "HUMAN_MESSAGE": (
                "Question: {input}\n\n"
                "{agent_scratchpad}\n\n")
        },
        "structured-chat-agent": {
            "SYSTEM_PROMPT": (
                "Respond to the human as helpfully and accurately as possible. You have access to the following tools:\n\n"
                "{tools}\n\n"
                "Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\n"
                'Valid "action" values: "Final Answer" or {tool_names}\n\n'
                "Provide only ONE action per $JSON_BLOB, as shown:\n\n"
                '```\n{{\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}\n```\n\n'
                "Follow this format:\n\n"
                "Question: input question to answer\n"
                "Thought: consider previous and subsequent steps\n"
                "Action:\n```\n$JSON_BLOB\n```\n"
                "Observation: action result\n"
                "... (repeat Thought/Action/Observation N times)\n"
                "Thought: I know what to respond\n"
                'Action:\n```\n{{\n  "action": "Final Answer",\n  "action_input": "Final response to human"\n}}\n\n'
                "Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation\n"
            ),
            "HUMAN_MESSAGE": (
                "{input}\n\n"
                "{agent_scratchpad}\n\n"
            )
            # '(reminder to respond in a JSON blob no matter what)')
        },
        "platform-agent": {
            "SYSTEM_PROMPT": (
                "You are a helpful assistant"
            ),
            "HUMAN_MESSAGE": (
                "{input}\n\n"
            )
        },
        "platform-knowledge-mode": {
            "SYSTEM_PROMPT": (
                "</think>You are ChatChat,  a content manager, you are familiar with how to find data from complex projects and better respond to users\n"
                "\n"
                "\n"
                "CRITICAL: TOOL RULES: All tool usage MUST ` Tool Use Formatting` the specified structured format. \n"
                "CRITICAL: THINKING RULES: In <thinking> tags, assess what information you already have and what information you need to proceed with the task. Include detailed output description text within <thinking> tags and always specify the `TOOL USE` next action to take.\n"
                "CRITICAL: MCP TOOL RULES: All MCP tool usage MUST strictly follow the Output Structure rules defined for `use_mcp_tool`. The output will always be returned within <use_mcp_tool> tags with the specified structured format.\n"
                "IMPORTANT: This tool usage process will be repeated multiple times throughout task completion. Each and every MCP tool call MUST follow the Output Structure rules without exception. The structured format must be applied consistently across all iterations to ensure proper parsing and execution.\n"
                "\n"
                "====\n"
                "\n"
                "TOOL USE\n"
                "You have access to a set of tools that are executed upon the user's approval. You can use one tool per message, and will receive the result of that tool use in the user's response. You use tools step-by-step to accomplish a given task, with each tool use informed by the result of the previous tool use.\n"
                "\n"
                "CRITICAL: MCP TOOL RULES: All MCP tool usage MUST strictly follow the Output Structure rules defined for `use_mcp_tool`. The output will always be returned within <use_mcp_tool> tags with the specified structured format.\n"
                "IMPORTANT: This tool usage process will be repeated multiple times throughout task completion. Each and every MCP tool call MUST follow the Output Structure rules without exception. The structured format must be applied consistently across all iterations to ensure proper parsing and execution.\n"
                "\n"
                "# Tool Use Formatting\n"
                "\n"
                "CRITICAL: TOOL USE FORMATTING: Tool use is formatted using XML-style tags. The tool name is enclosed in opening and closing tags, and each parameter is similarly enclosed within its own set of tags. This format is MANDATORY for proper parsing and execution. Here's the structure:\n"
                "\n"
                "<tool_name>\n"
                "<parameter1_name>value1</parameter1_name>\n"
                "<parameter2_name>value2</parameter2_name>\n"
                "...\n"
                "</tool_name>\n"
                "\n"
                "For example:\n"
                "\n"
                "<read_file>\n"
                "<path>src/main.js</path>\n"
                "</read_file>\n"
                "\n"
                "\n"
                "# Tools\n"
                "\n"
                "{tools}\n"
                "\n"
                "## use_mcp_tool\n"
                "Description: Request to use a tool provided by a connected MCP server. Each MCP server can provide multiple tools with different capabilities. Tools have defined input schemas that specify required and optional parameters.\n"
                "Parameters:\n"
                "- server_name: (required) The name of the MCP server providing the tool\n"
                "- tool_name: (required) The name of the tool to execute\n"
                "- arguments: (required) A JSON object containing the tool's input parameters, following the tool's input schema\n"
                "\n"
                "Usage:\n"
                "<use_mcp_tool>\n"
                "<server_name>server name here</server_name>\n"
                "<tool_name>tool name here</tool_name>\n"
                "<arguments>\n"
                "{{\n"
                "  \"param1\": \"value1\",\n"
                "  \"param2\": \"value2\"\n"
                "}}\n"
                "</arguments>\n"
                "</use_mcp_tool>\n"
                "\n"
                "Output Structure:\n"
                "The tool will return a structured response within <use_mcp_tool> tags containing:\n"
                "<use_mcp_tool>\n"
                "- success: boolean indicating if the tool execution succeeded\n"
                "- result: the actual output data from the tool execution\n"
                "- error: error message if the execution failed (null if successful)\n"
                "- server_name: the name of the MCP server that executed the tool\n"
                "- tool_name: the name of the tool that was executed\n"
                "</use_mcp_tool>\n"
                "\n"
                "\n"
                "## access_mcp_resource\n"
                "Description: Request to access a resource provided by a connected MCP server. Resources represent data sources that can be used as context, such as files, API responses, or system information.\n"
                "Parameters:\n"
                "- server_name: (required) The name of the MCP server providing the resource\n"
                "- uri: (required) The URI identifying the specific resource to access\n"
                "Usage:\n"
                "<access_mcp_resource>\n"
                "<server_name>server name here</server_name>\n"
                "<uri>resource URI here</uri>\n"
                "</access_mcp_resource>\n"
                "\n"
                "\n"
                "====\n"
                "\n"
                "# Tool Use Examples\n"
                "\n"
                "## Example 1: Requesting to use an MCP tool\n"
                "\n"
                "<use_mcp_tool>\n"
                "<server_name>weather-server</server_name>\n"
                "<tool_name>get_forecast</tool_name>\n"
                "<arguments>\n"
                "{{\n"
                "  \"city\": \"San Francisco\",\n"
                "  \"days\": 5\n"
                "}}\n"
                "</arguments>\n"
                "</use_mcp_tool>\n"
                "\n"
                "## Example 2: Requesting to access an MCP resource\n"
                "\n"
                "<access_mcp_resource>\n"
                "<server_name>weather-server</server_name>\n"
                "<uri>weather://san-francisco/current</uri>\n"
                "</access_mcp_resource>\n"
                "\n"
                "\n"
                "====\n"
                "\n"
                "MCP SERVERS\n"
                "\n"
                "The Model Context Protocol (MCP) enables communication between the system and locally running MCP servers that provide additional tools and resources to extend your capabilities.\n"
                "\n"
                "CRITICAL: MCP TOOL RULES: All MCP tool usage MUST strictly follow the Output Structure rules defined for `use_mcp_tool`. The output will always be returned within <use_mcp_tool> tags with the specified structured format.\n"
                "IMPORTANT: This tool usage process will be repeated multiple times throughout task completion. Each and every MCP tool call MUST follow the Output Structure rules without exception. The structured format must be applied consistently across all iterations to ensure proper parsing and execution.\n"
                "\n"
                "# Connected MCP Servers\n"
                "\n"
                "When a server is connected, you can use the server's tools via the `use_mcp_tool` tool, and access the server's resources via the `access_mcp_resource` tool.\n"
                "\n"
                "\n"
                "{mcp_tools}\n"
                "\n"
                "\n"
                "====\n"
                "\n"
                "\n"
                "# Choosing the Appropriate Tool\n"
                "\n"
                "None\n"
                "\n"
                "\n"
                "====\n"
                "# Auto-formatting Considerations\n"
                " \n"
                "None\n"
                "\n"
                "\n"
                "====\n"
                "# Workflow Tips\n"
                "\n"
                "None\n"
                "\n"
                "\n"
                "====\n"
                " \n"
                "CAPABILITIES\n"
                "\n"
                "- You have access to tools that\n"
                "\n"
                "- You have access to MCP servers that may provide additional tools and resources. Each server may provide different capabilities that you can use to accomplish tasks more effectively.\n"
                "\n"
                "\n"
                "====\n"
                "\n"
                "RULES\n"
                "\n"
                "CRITICAL: Always adhere to this format for the tool use to ensure proper parsing and execution. Before completing the user's final task, all intermediate tool usage processes must maintain proper parsing and execution. Each tool call must be correctly formatted and executed according to the specified XML structure to ensure successful task completion.\n"
                "CRITICAL: MCP TOOL RULES: 1. All MCP tool output must be enclosed within <use_mcp_tool> opening and closing tags without exception.\n"
                "CRITICAL: MCP TOOL RULES: 2. The structured response format must be strictly followed for proper parsing and execution.\n"
                "CRITICAL: MCP TOOL RULES: 3. Before completing user's final task, all intermediate MCP tool processes must maintain proper parsing and execution.\n"
                "CRITICAL: THINKING RULES: In <thinking> tags, assess what information you already have and what information you need to proceed with the task. Include detailed output description text within <thinking> tags and always specify the `TOOL USE` next action to take.\n"
                "CRITICAL: PARAMETER RULES: 1. ALL parameters marked as (required) MUST be provided with actual content - empty or null values are strictly forbidden.\n"
                "CRITICAL: PARAMETER RULES: 2. The 'uri' parameter MUST contain a valid resource URI string.\n"
                "CRITICAL: PARAMETER RULES: 3. Missing parameters or empty parameter values will cause resource access to fail.\n"
                "CRITICAL: PARAMETER RULES: 4. ALL parameters marked as (required) MUST be provided with actual content - empty or null values are strictly forbidden.\n"
                "CRITICAL: PARAMETER RULES: 5. The 'arguments' parameter MUST contain a valid JSON object with appropriate parameter values for the specified tool.\n"
                "CRITICAL: PARAMETER RULES: 6. Missing parameters or empty parameter values will cause tool execution to fail.\n"
                "CRITICAL: Tool Use RULES: 1. If multiple actions are needed, use one tool at a time per message to accomplish the task iteratively, with each tool use being informed by the result of the previous tool use. Do not assume the outcome of any tool use. Each step must be informed by the previous step's result.\n"
                "CRITICAL: Tool Use RULES: 2. Formulate your tool use using the XML format specified for each tool. by example `TOOL USE`\n"
                "Your current working directory is: {current_working_directory}\n"
                "You are STRICTLY FORBIDDEN from starting your messages with \"Great\", \"Certainly\", \"Okay\", \"Sure\". You should NOT be conversational in your responses, but rather direct and to the point. For example you should NOT say \"Great, I've find's the Chunk\" but instead something like \"I've find's the Chunk\". It is important you be clear and technical in your messages.\n"
                "When presented with images, utilize your vision capabilities to thoroughly examine them and extract meaningful information. Incorporate these insights into your thought process as you accomplish the user's task.\n"
                "At the end of each user message, you will automatically receive environment_details. This information is not written by the user themselves, but is auto-generated to provide potentially relevant context about the project structure and environment. While this information can be valuable for understanding the project context, do not treat it as a direct part of the user's request or response. Use it to inform your actions and decisions, but don't assume the user is explicitly asking about or referring to this information unless they clearly do so in their message. When using environment_details, explain your actions clearly to ensure the user understands, as they may not be aware of these details.\n"
                "MCP operations should be used one at a time, similar to other tool usage. Wait for confirmation of success before proceeding with additional operations.\n"

                "\n"
                "\n"
                "====\n"
                "\n"
                "SYSTEM INFORMATION\n"
                "\n"
                "None\n"
                "\n"
                "====\n"
                "\n"
                "OBJECTIVE\n"
                "\n"
                "You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.\n"
                "\n"
                "1. Analyze the user's task and set clear, achievable goals to accomplish it. Prioritize these goals in a logical order.\n"
                "2. Work through these goals sequentially, utilizing available tools one at a time as necessary. Each goal should correspond to a distinct step in your problem-solving process. You will be informed on the work completed and what's remaining as you go.\n"
                "3. Remember, you have extensive capabilities with access to a wide range of tools that can be used in powerful and clever ways as necessary to accomplish each goal. Before calling a tool, do some analysis within <thinking></thinking> tags. First, analyze the file structure provided in environment_details to gain context and insights for proceeding effectively. Then, think about which of the provided tools is the most relevant tool to accomplish the user's task.\n"
                "4. The user may provide feedback, which you can use to make improvements and try again. But DO NOT continue in pointless back and forth conversations, i.e. don't end your responses with questions or offers for further assistance.\n"
            ),
            "HUMAN_MESSAGE": (
                "{input}\n\n"
                "<environment_details>\n"
                "# Current Time\n"
                "{datetime}\n"
                "</environment_details>\n"
            )
        },
    }
    """Agent 模板"""

    postprocess_model: dict = {
        "default": "{{input}}",
    }
    """后处理模板"""



basic_settings = BasicSettings()
kb_settings = KBSettings()
platform_config = PlatformConfig()
api_model_settings = ApiModelSettings()
tool_settings = ToolSettings()
prompt_settings = PromptSettings()
