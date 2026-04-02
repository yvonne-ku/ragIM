from server.settings import kb_settings
from server.utils import logger
from server import settings
from server.kb_service.chromadb_service import get_kb
from server.utils import StaticPathTools

def init_backend(
        plat_form_url: str = settings.platform_config.api_base_url,
        llm_model: str = settings.api_model_settings.DEFAULT_LLM_MODEL,
        embedding_model: str = settings.api_model_settings.DEFAULT_EMBEDDING_MODEL,
        recreate_kb: bool = False,
        kb_name: str = settings.api_model_settings.DEFAULT_KB_NAME,
):
    """初始化项目后端"""
    bs = settings.basic_settings
    ks = settings.kb_settings

    # 1. 初始化项目目录结构
    logger.info(f"开始初始化项目目录结构，项目根目录：{bs.RAGIM_ROOT}")
    bs.make_dirs()
    logger.info("项目目录结构初始化完成")

    # 2. 初始化知识库数据库表
    # logger.info(f"开始初始化知识库数据库，数据库目录：{bs.SQLALCHEMY_DATABASE_URI}")
    # ks.create_tables()
    # logger.info("初始化知识库数据库成功")

    # 3. 启动参数覆盖默认配置
    logger.info("开始将启动参数覆盖默认配置")
    if plat_form_url:
        settings.platform_config.api_base_url = plat_form_url
    if llm_model:
        settings.api_model_settings.DEFAULT_LLM_MODEL = llm_model
    if embedding_model:
        settings.api_model_settings.DEFAULT_EMBEDDING_MODEL = embedding_model
    if kb_name:
        settings.api_model_settings.DEFAULT_KB_NAME = kb_name
    logger.info("启动参数覆盖默认配置完成")

    # 4. 构建向量库
    if recreate_kb:
        # 初始化知识库服务
        kb = get_kb(kb_name=kb_name)
        # 删除原有库
        try:
            kb.delete_collection()
            logger.info(f"已删除原有知识库：{kb_name}")
        except Exception as e:
            logger.warning(f"删除原有知识库失败或知识库不存在: {e}")      

        # 确定原始数据存在
        raw_data_path = StaticPathTools.get_raw_path(kb_name)
        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
            logger.warning(f"原始数据目录 {raw_data_path} 不存在，已创建。请将文档放入此目录后再重试。")
            return
        # 遍历文件夹导入原始文件
        files = [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]
        if not files:
            logger.warning(f"原始数据目录 {raw_data_path} 下没有找到任何文件。")
            return
        logger.info(f"检测到 {len(files)} 个待处理文件，开始导入知识库 {kb_name}...")

        # 遍历处理文件
        for file_name in files:
            kb.add_files(file_name)
            logger.info(f"文件 {file_name} 导入并向量化成功")
        logger.info(f"知识库 {kb_name} 初始化/重建完成")

