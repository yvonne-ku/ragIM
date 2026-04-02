import sys
from loguru import logger
from pathlib import Path
import os

from server import settings

logger.remove()  # 移除默认配置
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    diagnose=True,
    backtrace=True
)

__all__ = ["logger", "StaticPathTools"]

class StaticPathTools:
    @staticmethod
    def get_kb_path(kb_name: str):
        """知识库根目录"""
        return os.path.join(settings.basic_settings.KB_ROOT, kb_name)
        
    @staticmethod
    def get_raw_path(kb_name: str):
        """知识库 raw data 地址"""
        return os.path.join(StaticPathTools.get_kb_path(kb_name), "content")

    @staticmethod
    def get_vs_path(kb_name: str, vector_name: str):
        """知识库 vector 地址"""
        return os.path.join(StaticPathTools.get_kb_path(kb_name), "vector_store", vector_name)
    
    @staticmethod
    def get_full_path(kb_name: str, doc_name: str):
        # 检查是够为绝对路径，win 上的 \\ 格式
        if os.path.isabs(doc_name) or (doc_name.startswith('/') and ':' not in doc_name):
            if '/' in doc_name and '\\' not in doc_name:
                if len(doc_name) > 2 and doc_name[1] == '/':
                    doc_name = doc_name.replace('/', '\\')
            return doc_name

        doc_path = Path(StaticPathTools.get_doc_path(kb_name)).resolve()    # 转化为绝对路径
        full_path = (doc_path / doc_name).resolve()
        if str(full_path).startswith(str(doc_path)):
            return str(full_path)
