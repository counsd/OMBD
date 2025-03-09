import sys
from datetime import datetime

from loguru import logger as _logger

from app.config import PROJECT_ROOT


_print_level = "INFO"


def define_log_level(print_level="INFO", logfile_level="DEBUG", name: str = None):
    """调整日志级别到指定级别以上"""
    global _print_level
    _print_level = print_level

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d%H%M%S")
    log_name = (
        f"{name}_{formatted_date}" if name else formatted_date
    )  # 使用前缀名称命名日志

    _logger.remove()
    _logger.add(sys.stderr, level=print_level)
    _logger.add(PROJECT_ROOT / f"logs/{log_name}.log", level=logfile_level)
    return _logger


logger = define_log_level()


if __name__ == "__main__":
    logger.info("启动应用程序")
    logger.debug("调试消息")
    logger.warning("警告消息")
    logger.error("错误消息")
    logger.critical("严重消息")

    try:
        raise ValueError("测试错误")
    except Exception as e:
        logger.exception(f"发生错误: {e}")