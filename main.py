import asyncio

from app.agent.manus import Manus
from app.logger import logger


async def main():
    agent = Manus()
    while True:
        try:
            prompt = input("输入您的提示（输入 'exit' 退出）：")
            if prompt.lower() == "exit":
                logger.info("再见！")
                break
            if prompt.strip().isspace():
                logger.warning("跳过空提示。")
                continue
            logger.warning("正在处理您的请求...")
            await agent.run(prompt)
        except KeyboardInterrupt:
            logger.warning("再见！")
            break


if __name__ == "__main__":
    asyncio.run(main())