from typing import List

import cv2
import numpy as np
import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from PIL import Image

# from ragim.settings import Settings
from ragim.ocr_loader.ocr import get_ocr


class RapidOCRPDFLoader(UnstructuredFileLoader):

    """
    自定义PDF文档加载器（继承LangChain通用加载器）
    核心功能：
    1. 提取PDF页面原生文字（非图片文字）
    2. 提取PDF中的图片，过滤小图片后做OCR识别
    3. 处理PDF页面旋转问题，自动校正图片角度
    4. 拼接所有文字，返回LangChain兼容的文档格式
    """
    def _get_elements(self) -> List:
        def rotate_img(img, angle):
            """
            旋转图片（解决PDF页面旋转导致的图片方向错误问题）
            :param img: 输入的图片numpy数组（CV2格式）
            :param angle: 旋转角度（正值=逆时针，负值=顺时针）
            :return: 旋转后的图片numpy数组
            """
            h, w = img.shape[:2]
            rotate_center = (w / 2, h / 2)
            # 获取旋转矩阵
            # 参数1为旋转中心点;
            # 参数2为旋转角度,正值-逆时针旋转;负值-顺时针旋转
            # 参数3为各向同性的比例因子,1.0原图，2.0变成原来的2倍，0.5变成原来的0.5倍
            M = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)
            # 计算图像新边界
            new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
            new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))
            # 调整旋转矩阵以考虑平移
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2

            rotated_img = cv2.warpAffine(img, M, (new_w, new_h))
            return rotated_img


        """
        PDF转文字核心函数：
        1. 提取PDF每页的原生文字
        2. 提取每页图片，过滤小图片后OCR识别
        3. 处理图片旋转，保证OCR准确性
        :param filepath: PDF文件路径
        :return: 拼接后的所有文字（原生+OCR）
        """
        def pdf2text(filepath):
            import fitz  # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            import numpy as np
            
            ocr = get_ocr()
            doc = fitz.open(filepath)
            resp = ""

            # 初始化进度条
            b_unit = tqdm.tqdm(
                total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0"
            )
            # 遍历每一页
            for i, page in enumerate(doc):
                b_unit.set_description(
                    "RapidOCRPDFLoader context page index: {}".format(i)
                )
                b_unit.refresh()

                # 提取原生文字
                text = page.get_text("")
                resp += text + "\n"

                # 提取图片信息
                img_list = page.get_image_info(xrefs=True)
                for img in img_list:
                    # 获取图片的 xref 和 bbox 信息
                    if xref := img.get("xref"):
                        bbox = img["bbox"]

                        # 1. 过滤小图片：避免对图标/水印等无效小图做OCR识别
                        img_width_ratio = (bbox[2] - bbox[0]) / page.rect.width     # 图片宽度占页面比例
                        img_height_ratio = (bbox[3] - bbox[1]) / page.rect.height   # 图片高度占页面比例
                        
                        # 在调整 settings 文件之前先采用硬编码
                        PDF_OCR_WIDTH_THRESHOLD = 0.05  # 宽度占页面比例阈值（5%）
                        PDF_OCR_HEIGHT_THRESHOLD = 0.05  # 高度占页面比例阈值（5%）
                        if img_width_ratio < PDF_OCR_WIDTH_THRESHOLD or img_height_ratio < PDF_OCR_HEIGHT_THRESHOLD:
                            continue

                        # 对比配置中的阈值：任意维度比例低于阈值则跳过
                        # if img_width_ratio < Settings.kb_settings.PDF_OCR_THRESHOLD[0] or \
                        #     img_height_ratio < Settings.kb_settings.PDF_OCR_THRESHOLD[1]:
                        #     continue

                        # 根据 xref 获取图片像素数据
                        pix = fitz.Pixmap(doc, xref)
                        samples = pix.samples

                        # 2. 处理 PDF 页面旋转：
                        # 如果有旋转角度，需要旋转图片
                        # 如果无旋转，直接将图片转为numpy数组
                        if int(page.rotation) != 0:  # 如果Page有旋转角度，则旋转图片
                            img_array = np.frombuffer(
                                pix.samples, dtype=np.uint8
                            ).reshape(pix.height, pix.width, -1)
                            tmp_img = Image.fromarray(img_array)
                            ori_img = cv2.cvtColor(np.array(tmp_img), cv2.COLOR_RGB2BGR)
                            rot_img = rotate_img(img=ori_img, angle=360 - page.rotation)
                            img_array = cv2.cvtColor(rot_img, cv2.COLOR_RGB2BGR)
                        else:
                            img_array = np.frombuffer(
                                pix.samples, dtype=np.uint8
                            ).reshape(pix.height, pix.width, -1)

                        # 3. 进行 OCR 识别，合并结果
                        result, _ = ocr(img_array)
                        if result:
                            ocr_result = [line[1] for line in result]
                            resp += "\n".join(ocr_result)

                # 更新进度
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)

        return [Document(
            page_content=text.strip(),  # 提取的纯文字（去除首尾空格）
            metadata={
                "source": self.file_path,  # 文件路径（溯源用）
                "file_type": "pdf",        # 标记文件类型（便于后续处理）
                "processed_by": "RapidOCRPDFLoader"  # 标记处理加载器
            }
        )]
        # from unstructured.partition.text import partition_text
        # return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    from pathlib import Path
    file_path = Path(__file__).parent.parent.parent / "test" / "samples" / "ocr_test.pdf"
    loader = RapidOCRPDFLoader(file_path=str(file_path))
    docs = loader.load()
    print(docs)