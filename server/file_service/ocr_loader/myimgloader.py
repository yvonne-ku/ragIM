from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from server.file_service.ocr_loader.ocr import get_ocr


class RapidOCRLoader(UnstructuredFileLoader):

    """
    重写 _get_elements 方法，实现对图片的解析
    """
    def _get_elements(self) -> List:
        
        """
        对图片进行 OCR 识别，返回识别结果
        """
        def img2text(filepath):
            resp = ""
            ocr = get_ocr()
            result, _ = ocr(filepath)
            if result:
                ocr_result = [line[1] for line in result]
                resp += "\n".join(ocr_result)
            return resp

        text = img2text(self.file_path)
        # return [Document(page_content=text.strip(), metadata={"source": self.file_path})]


        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    from pathlib import Path
    file_path = Path(__file__).parent.parent.parent / "test" / "samples" / "ocr_test.jpg"
    loader = RapidOCRLoader(file_path=str(file_path))
    docs = loader.load()
    print(docs)