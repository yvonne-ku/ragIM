
from server.utils import logger
from server import settings

def init_backend():
    """初始化项目后端"""
    bs = settings.basic_settings

    # 初始化项目目录结构
    logger.info(f"开始初始化项目目录结构，项目根目录：{bs.RAGIM_ROOT}")
    bs.make_dirs()
    logger.info("项目目录结构初始化完成")

    # 检查知识库文件是否在目标位置，不存在就复制到目标位置
    if bs.