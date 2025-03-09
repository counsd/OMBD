import json
from typing import Any, List, Literal

from pydantic import Field

from app.agent.react import ReActAgent
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message, ToolCall
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "需要工具调用但未提供"


class ToolCallAgent(ReActAgent):
    """用于处理工具/函数调用的代理基类，具有增强的抽象性"""

    name: str = "toolcall"
    description: str = "一个可以执行工具调用的代理。"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: Literal["none", "auto", "required"] = "auto"
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)

    max_steps: int = 30

    async def think(self) -> bool:
        """处理当前状态并使用工具决定下一步操作"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        # 获取带有工具选项的响应
        response = await self.llm.ask_tool(
            messages=self.messages,
            system_msgs=[Message.system_message(self.system_prompt)]
            if self.system_prompt
            else None,
            tools=self.available_tools.to_params(),
            tool_choice=self.tool_choices,
        )
        self.tool_calls = response.tool_calls

        # 记录响应信息
        logger.info(f"✨ {self.name} 的思考：{response.content}")
        logger.info(
            f"🛠️ {self.name} 选择了 {len(response.tool_calls) if response.tool_calls else 0} 个工具来使用"
        )
        if response.tool_calls:
            logger.info(
                f"🧰 正在准备的工具：{[call.function.name for call in response.tool_calls]}"
            )

        try:
            # 处理不同的 tool_choices 模式
            if self.tool_choices == "none":
                if response.tool_calls:
                    logger.warning(
                        f"🤔 嗯，{self.name} 尝试在工具不可用时使用工具！"
                    )
                if response.content:
                    self.memory.add_message(Message.assistant_message(response.content))
                    return True
                return False

            # 创建并添加助手消息
            assistant_msg = (
                Message.from_tool_calls(
                    content=response.content, tool_calls=self.tool_calls
                )
                if self.tool_calls
                else Message.assistant_message(response.content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == "required" and not self.tool_calls:
                return True  # 将在 act() 中处理

            # 对于 'auto' 模式，如果存在内容但没有命令，则继续
            if self.tool_choices == "auto" and not self.tool_calls:
                return bool(response.content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 哎呀！{self.name} 的思考过程出了问题：{e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"处理时遇到错误：{str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """执行工具调用并处理它们的结果"""
        if not self.tool_calls:
            if self.tool_choices == "required":
                raise ValueError(TOOL_CALL_REQUIRED)

            # 如果没有工具调用，则返回最后一条消息的内容
            return self.messages[-1].content or "没有内容或命令可执行"

        results = []
        for command in self.tool_calls:
            result = await self.execute_tool(command)
            logger.info(
                f"🎯 工具 '{command.function.name}' 完成了它的任务！结果：{result}"
            )

            # 将工具响应添加到记忆中
            tool_msg = Message.tool_message(
                content=result, tool_call_id=command.id, name=command.function.name
            )
            self.memory.add_message(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """执行单个工具调用并进行健壮的错误处理"""
        if not command or not command.function or not command.function.name:
            return "错误：无效的命令格式"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"错误：未知工具 '{name}'"

        try:
            # 解析参数
            args = json.loads(command.function.arguments or "{}")

            # 执行工具
            logger.info(f"🔧 正在激活工具：'{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # 格式化结果以便显示
            observation = (
                f"观察到执行命令 `{name}` 的输出：\n{str(result)}"
                if result
                else f"命令 `{name}` 已完成，没有输出"
            )

            # 处理特殊工具如 `finish`
            await self._handle_special_tool(name=name, result=result)

            return observation
        except json.JSONDecodeError:
            error_msg = f"解析 {name} 的参数时出错：JSON 格式无效"
            logger.error(
                f"📝 哎呀！'{name}' 的参数有问题 - JSON 无效，参数：{command.function.arguments}"
            )
            return f"错误：{error_msg}"
        except Exception as e:
            error_msg = f"⚠️ 工具 '{name}' 遇到问题：{str(e)}"
            logger.error(error_msg)
            return f"错误：{error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """处理特殊工具执行和状态变化"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # 将代理状态设置为已完成
            logger.info(f"🏁 特殊工具 '{name}' 已完成任务！")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """确定工具执行是否应完成代理"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """检查工具名称是否在特殊工具列表中"""
        return name.lower() in [n.lower() for n in self.special_tool_names]