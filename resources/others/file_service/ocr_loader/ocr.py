from typing import TYPE_CHECKING

# TYPE_CHECKING 是一个特殊的布尔常量
# 仅在「代码静态类型检查 / 编辑器语法提示」时为 True，
# 程序实际运行时永远为 False。
# 没有 rapidocr_paddle 时，使用 rapidocr_onnxruntime
# 如果你的电脑没有 NVIDA 显卡（不支持 CUDA），使用 rapidocr_paddle 时，
# 否则使用 CPU 版本，使用 rapidocr_onnxruntime
if TYPE_CHECKING:
    try:
        from rapidocr_paddle import RapidOCR
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR


def get_ocr(use_cuda: bool = True) -> "RapidOCR":
    try:
        from rapidocr_paddle import RapidOCR
        ocr = RapidOCR(
            det_use_cuda=use_cuda, cls_use_cuda=use_cuda, rec_use_cuda=use_cuda
        )
    except ImportError:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    return ocr
