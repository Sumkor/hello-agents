# -*- coding: utf-8 -*-
"""
AgentScope 极简示例：两个 Agent 通过 MsgHub 对话
"""
import asyncio
import os

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.pipeline import MsgHub
from agentscope.formatter import DashScopeMultiAgentFormatter


async def main():
    # 1. 检查 API Key
    if "DASHSCOPE_API_KEY" not in os.environ:
        print("请设置环境变量 DASHSCOPE_API_KEY")
        return

    # 2. 共享模型配置
    model = DashScopeChatModel(
        model_name="qwen-max",
        api_key=os.environ["DASHSCOPE_API_KEY"],
    )

    # 3. 创建两个 Agent
    alice = ReActAgent(
        name="Alice",
        sys_prompt="你是一个热爱诗歌的作家，喜欢用优美的语言表达观点。",
        model=model,
        formatter=DashScopeMultiAgentFormatter(),
    )
    bob = ReActAgent(
        name="Bob",
        sys_prompt="你是一个理性严谨的科学家，习惯用逻辑和数据说话。",
        model=model,
        formatter=DashScopeMultiAgentFormatter(),
    )

    # 4. 通过 MsgHub 让两人对话
    print("=" * 50)
    print("Alice 和 Bob 正在讨论「人工智能能否创作真正的艺术」")
    print("=" * 50)

    async with MsgHub(
        participants=[alice, bob],
        announcement=Msg(name="System", content="请讨论：人工智能能否创作真正的艺术？", role="system"),
    ) as hub:
        for turn in range(3):  # 三轮对话
            for agent in [alice, bob]:
                resp = await agent()
                # if resp:
                #     print(f"\n[{agent.name}]: {resp.content}")

    print("\n" + "=" * 50)
    print("对话结束")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
