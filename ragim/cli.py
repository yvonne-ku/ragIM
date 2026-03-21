import click
from pathlib import Path
import shutil
import typing as t

from chatchat.startup import main as startup_main
from chatchat.init_database import main as kb_main, create_tables, folder2db
from chatchat.settings import Settings
from chatchat.utils import build_logger
from chatchat.server.utils import get_default_embedding


logger = build_logger()


# 定义一个命令组，在组里定义多个子命令，每个子命令对应一个函数，实现 CLI 的不同功能
@click.group(help="CLI tools for ragIM.")
def main():
    ...

# 定义子命令 init 及其选项
# -x 是短选项名
# --xinference-endpoint 是长选项名
# xf_endpoint 是指当用户输入这个选项后，值会赋值给 init 函数的 xf_endpoint 参数
# help 描述了该选项的作用和默认值
@main.command("init", help="initialize the ragIM.")
@click.option("-x", "--xinference-endpoint", "xf_endpoint",
              help="指定 Xinference API 服务地址。默认为 http://127.0.0.1:9997/v1")
@click.option("-l", "--llm-model",
              help="指定默认 LLM 模型。默认为 glm4-chat")
@click.option("-e", "--embed-model",
              help="指定默认 Embedding 模型。默认为 bge-large-zh-v1.5")
@click.option("-r", "--recreate-kb",
              is_flag=True,
              show_default=True,
              default=False,
              help="同时重建知识库（必须确保指定的 embed model 可用）。")
@click.option("-k", "--kb-names", "kb_names",
              show_default=True,
              default="samples",
              help="要重建知识库的名称。可以指定多个知识库名称，以 , 分隔。")
def init(
    xf_endpoint: str = "",
    llm_model: str = "",
    embed_model: str = "",
    recreate_kb: bool = False,
    kb_names: str = "",
):
    Settings.set_auto_reload(False) # 关闭配置文件的自动重载，接下来要根据传参修改配置信息
    bs = Settings.basic_settings  # 拿到配置文件中的基本配置类
    kb_names = [x.strip() for x in kb_names.split(",")] # 解析知识库名称

    # 创建预定义的项目目录结构
    logger.success(f"开始初始化项目数据目录：{Settings.CHATCHAT_ROOT}")
    bs.make_dirs()
    logger.success("创建所有数据目录：成功。")

    # 检查示例知识库文件是否存在于目标位置，不存在就复制过去
    if(bs.PACKAGE_ROOT / "data/knowledge_base/samples" != Path(bs.KB_ROOT_PATH) / "samples"):
        shutil.copytree(bs.PACKAGE_ROOT / "data/knowledge_base/samples",
                        Path(bs.KB_ROOT_PATH) / "samples",
                        dirs_exist_ok=True)
    logger.success("复制 samples 知识库文件：成功。")

    # 初始化所需的数据库表
    create_tables()
    logger.success("初始化知识库数据库：成功。")

    # 如果指定了 API 的地址，设置到配置变量中
    if xf_endpoint:
        Settings.model_settings.MODEL_PLATFORMS[0].api_base_url = xf_endpoint
    # 如果指定了模型，设置到配置变量中
    if llm_model:
        Settings.model_settings.DEFAULT_LLM_MODEL = llm_model
    # 如果指定了嵌入模型，设置到配置变量中
    if embed_model:
        Settings.model_settings.DEFAULT_EMBEDDING_MODEL = embed_model

    Settings.create_all_templates()
    Settings.set_auto_reload(True)
    logger.success("生成默认配置文件：成功。")
    logger.success("请先检查确认 model_settings.yaml 里模型平台、LLM模型和 Embed模型 信息已经正确")

    # 如果指定了重新创建向量库，调用 folder2db 函数重建知识库
    if recreate_kb:
        folder2db(kb_names=kb_names,
                  mode="recreate_vs",
                  vs_type=Settings.kb_settings.DEFAULT_VS_TYPE,
                  embed_model=get_default_embedding())
        logger.success("<green>所有初始化已完成，执行 chatchat start -a 启动服务。</green>")
    else:
        logger.success("执行 ragim kb -r 初始化知识库，然后 ragim start -a 启动服务。")


main.add_command(startup_main, "start")
main.add_command(kb_main, "kb")


if __name__ == "__main__":
    main()
