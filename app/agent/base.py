from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """管理代理状态和执行的抽象基类。

    提供状态转换、内存管理和基于步骤的执行循环的基础功能。子类必须实现 `step` 方法。
    """

    # 核心属性
    name: str = Field(..., description="代理的唯一名称")
    description: Optional[str] = Field(None, description="可选的代理描述")

    # 提示
    system_prompt: Optional[str] = Field(
        None, description="系统级指令提示"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="确定下一步操作的提示"
    )

    # 依赖
    llm: LLM = Field(default_factory=LLM, description="语言模型实例")
    memory: Memory = Field(default_factory=Memory, description="代理的内存存储")
    state: AgentState = Field(
        default=AgentState.IDLE, description="当前代理状态"
    )

    # 执行控制
    max_steps: int = Field(default=10, description="终止前的最大步骤数")
    current_step: int = Field(default=0, description="执行中的当前步骤")

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # 允许子类中存在额外字段以提高灵活性

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """如果未提供，则用默认设置初始化代理。"""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """用于安全代理状态转换的上下文管理器。

        参数：
            new_state: 在上下文中转换到的状态。

        产出：
            None: 允许在新状态下执行。

        异常：
            ValueError: 如果 new_state 无效。
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"无效的状态: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # 在失败时转换到 ERROR 状态
            raise e
        finally:
            self.state = previous_state  # 恢复到之前的状态

    def update_memory(
        self,
        role: Literal["user", "system", "assistant", "tool"],
        content: str,
        **kwargs,
    ) -> None:
        """将消息添加到代理的内存中。

        参数：
            role: 消息发送者的角色（用户、系统、助手、工具）。
            content: 消息内容。
            **kwargs: 额外参数（例如，工具消息的 tool_call_id）。

        异常：
            ValueError: 如果角色不受支持。
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"不支持的消息角色: {role}")

        msg_factory = message_map[role]
        msg = msg_factory(content, **kwargs) if role == "tool" else msg_factory(content)
        self.memory.add_message(msg)

    async def run(self, request: Optional[str] = None) -> str:
        """异步执行代理的主要循环。

        参数：
            request: 可选的初始用户请求以进行处理。

        返回：
            总结执行结果的字符串。

        异常：
            RuntimeError: 如果代理在启动时不在空闲状态。
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"无法从状态 {self.state} 启动代理")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"执行步骤 {self.current_step}/{self.max_steps}")
                step_result = await self.step()

                # 检查是否卡住
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"步骤 {self.current_step}: {step_result}")

            if self.current_step >= self.max_steps:
                results.append(f"已终止: 达到最大步骤数 ({self.max_steps})")

        return "\n".join(results) if results else "未执行任何步骤"

    @abstractmethod
    async def step(self) -> str:
        """执行代理工作流中的单个步骤。

        必须由子类实现以定义具体行为。
        """

    def handle_stuck_state(self):
        """通过添加提示来处理卡住状态，以改变策略"""
        stuck_prompt = "\
        观察到重复响应。考虑新策略，避免重复尝试已无效的路径。"
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"代理检测到卡住状态。已添加提示: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """通过检测重复内容检查代理是否陷入循环"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # 统计相同内容的出现次数
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """从代理的内存中检索消息列表。"""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """设置代理内存中的消息列表。"""
        self.memory.messages = value