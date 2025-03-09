from typing import Dict, List, Literal, Optional, Union

from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.config import LLMSettings, config
from app.logger import logger  # 假设你的应用中已经设置了日志记录器
from app.schema import Message


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(self, config_name: str = "default", llm_config: Optional[LLMSettings] = None):
        if not hasattr(self, "client"):  # 只有在未初始化时才进行初始化
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url
            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def format_messages(messages: List[Union[dict, Message]]) -> List[dict]:
        """
        将消息格式化为 LLM 所需的 OpenAI 消息格式。

        参数：
            messages: 可以是字典或 Message 对象的消息列表

        返回：
            List[dict]: 以 OpenAI 格式格式化的消息列表

        异常：
            ValueError: 如果消息无效或缺少必填字段
            TypeError: 如果提供了不受支持的消息类型

        示例：
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # 如果消息已经是字典，确保它包含必填字段
                if "role" not in message:
                    raise ValueError("消息字典必须包含 'role' 字段")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # 如果消息是 Message 对象，将其转换为字典
                formatted_messages.append(message.to_dict())
            else:
                raise TypeError(f"不支持的消息类型: {type(message)}")

        # 验证所有消息都包含必填字段
        for msg in formatted_messages:
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValueError(f"无效的角色: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "消息必须包含 'content' 或 'tool_calls' 之一"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        向 LLM 发送提示并获取响应。

        参数：
            messages: 对话消息列表
            system_msgs: 可选的系统消息列表，用于前置
            stream (bool): 是否以流式方式获取响应
            temperature (float): 响应的采样温度

        返回：
            str: 生成的响应

        异常：
            ValueError: 如果消息无效或响应为空
            OpenAIError: 如果 API 调用在重试后失败
            Exception: 对于意外错误
        """
        try:
            # 格式化系统和用户消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            if not stream:
                # 非流式请求
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=temperature or self.temperature,
                    stream=False,
                )
                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("LLM 返回了空响应")
                return response.choices[0].message.content

            # 流式请求
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=temperature or self.temperature,
                stream=True,
            )

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # 流式传输后换行
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("流式 LLM 返回了空响应")
            return full_response

        except ValueError as ve:
            logger.error(f"验证错误: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API 错误: {oe}")
            raise
        except Exception as e:
            logger.error(f"ask 中发生意外错误: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 60,
        tools: Optional[List[dict]] = None,
        tool_choice: Literal["none", "auto", "required"] = "auto",
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        使用函数/工具向 LLM 发送请求并返回响应。

        参数：
            messages: 对话消息列表
            system_msgs: 可选的系统消息列表，用于前置
            timeout: 请求超时时间（秒）
            tools: 要使用的工具列表
            tool_choice: 工具选择策略
            temperature: 响应的采样温度
            **kwargs: 额外的完成参数

        返回：
            ChatCompletionMessage: 模型的响应

        异常：
            ValueError: 如果工具、tool_choice 或消息无效
            OpenAIError: 如果 API 调用在重试后失败
            Exception: 对于意外错误
        """
        try:
            # 验证 tool_choice
            if tool_choice not in ["none", "auto", "required"]:
                raise ValueError(f"无效的 tool_choice: {tool_choice}")

            # 格式化消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # 如果提供了工具，进行验证
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("每个工具必须是一个包含 'type' 字段的字典")

            # 设置完成请求
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                timeout=timeout,
                **kwargs,
            )

            # 检查响应是否有效
            if not response.choices or not response.choices[0].message:
                print(response)
                raise ValueError("LLM 返回了无效或空响应")

            return response.choices[0].message

        except ValueError as ve:
            logger.error(f"ask_tool 中的验证错误: {ve}")
            raise
        except OpenAIError as oe:
            if isinstance(oe, AuthenticationError):
                logger.error("身份验证失败。请检查 API 密钥。")
            elif isinstance(oe, RateLimitError):
                logger.error("速率限制超出。考虑增加重试次数。")
            elif isinstance(oe, APIError):
                logger.error(f"API 错误: {oe}")
            raise
        except Exception as e:
            logger.error(f"ask_tool 中发生意外错误: {e}")
            raise