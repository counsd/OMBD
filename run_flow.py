import asyncio
import time

from app.agent.manus import Manus
from app.flow.base import FlowType
from app.flow.flow_factory import FlowFactory
from app.logger import logger


async def run_flow():
    agents = {
        "manus": Manus(),
    }

    while True:
        try:
            prompt = input("输入您的提示（输入 'exit' 退出）：")
            if prompt.lower() == "exit":
                logger.info("再见！")
                break

            flow = FlowFactory.create_flow(
                flow_type=FlowType.PLANNING,
                agents=agents,
            )
            if prompt.strip().isspace():
                logger.warning("跳过空提示。")
                continue
            logger.warning("正在处理您的请求...")

            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    flow.execute(prompt),
                    timeout=3600  # 整个执行过程的超时时间为60分钟
                )
                elapsed_time = time.time() - start_time
                logger.info(f"请求在 {elapsed_time:.2f} 秒内处理完成")
                logger.info(result)
            except asyncio.TimeoutError:
                logger.error("请求处理在1小时后超时")
                logger.info("由于超时，操作已终止。请尝试更简单的请求。")

        except KeyboardInterrupt:
            logger.info("用户取消了操作。")
        except Exception as e:
            logger.error(f"错误：{str(e)}")


if __name__ == "__main__":
    asyncio.run(run_flow())