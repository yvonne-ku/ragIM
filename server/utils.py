from pathlib import Path
import os

from server.settings import Settings


class StaticPathTools:
    @staticmethod
    def get_kb_path(kb_name: str):
        return os.path.join(Settings.basic_settings.KB_ROOT_PATH, kb_name)
        
    @staticmethod
    def get_doc_path(kb_name: str):
        return os.path.join(StaticPathTools.get_kb_path(kb_name), "content")

    @staticmethod
    def get_vs_path(kb_name: str, vector_name: str):
        return os.path.join(StaticPathTools.get_kb_path(kb_name), "vector_store", vector_name)
    
    @staticmethod
    def get_full_path(kb_name: str, doc_name: str):
        doc_path = Path(StaticPathTools.get_doc_path(kb_name)).resolve()    # 转化为绝对路径
        full_path = (doc_path / doc_name).resolve()
        if str(full_path).startswith(str(doc_path)):
            return str(full_path)
