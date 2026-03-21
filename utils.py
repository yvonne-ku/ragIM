import sys
from loguru import logger

# 1. 清除默认配置
logger.remove()

# 2. 配置控制台输出
# 格式：时间 | 级别 | 文件名:函数名:行号 - 消息内容
logger.add(
    sys.stdout, 
    colorize=True, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    diagnose=True,
    backtrace=True
)

# 3. 可选：配置日志文件保存（自动滚动，每10MB换一个文件）
# log_path = os.path.join(os.getcwd(), "logs", "chatchat.log")
# logger.add(log_path, rotation="10 MB", encoding="utf-8", level="DEBUG")

# 导出 logger 供全局使用
__all__ = ["logger"]
