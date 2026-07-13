# test_react_agent_v2.py
import pytest
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry
from my_react_agent import MyReActAgent

# 加载环境变量
load_dotenv()


@pytest.fixture(scope="module")
def llm():
    return HelloAgentsLLM()


@pytest.fixture(scope="module")
def tool_registry():
    registry = ToolRegistry()
    # 注册计算器工具
    try:
        from hello_agents import calculate
        registry.register_function("calculate", "执行数学计算，支持基本的四则运算", calculate)
    except ImportError:
        pass
    # 注册搜索工具
    try:
        from hello_agents import search
        registry.register_function("search", "搜索互联网信息", search)
    except ImportError:
        pass
    return registry


@pytest.fixture
def react_agent(llm, tool_registry):
    return MyReActAgent(
        name="我的推理行动助手",
        llm=llm,
        tool_registry=tool_registry,
        max_steps=5
    )


@pytest.fixture
def custom_agent(llm):
    registry = ToolRegistry()
    try:
        from hello_agents import calculate
        registry.register_function("calculate", "数学计算工具", calculate)
    except ImportError:
        pass

    custom_prompt = """你是一个数学专家AI助手。

可用工具：{tools}

请按以下格式回应：
Thought: [你的思考]
Action: [tool_name[input] 或 Finish[答案]]

问题：{question}
历史：{history}

开始："""

    return MyReActAgent(
        name="数学专家助手",
        llm=llm,
        tool_registry=registry,
        max_steps=3,
        custom_prompt=custom_prompt
    )


def test_math_calculation(react_agent):
    """测试数学计算"""
    math_question = "请帮我计算：(25 + 15) * 3 - 8 的结果是多少？"
    result = react_agent.run(math_question)
    print(f"\n数学计算响应: {result}")
    assert result is not None
    assert len(result) > 0


def test_search_question(react_agent):
    """测试信息搜索"""
    search_question = "Python编程语言是什么时候发布的？请告诉我具体的年份。"
    result = react_agent.run(search_question)
    print(f"\n信息搜索响应: {result}")
    assert result is not None
    assert len(result) > 0


def test_complex_reasoning(react_agent):
    """测试复合推理"""
    complex_question = "如果一个班级有30个学生，其中60%是女生，那么男生有多少人？请先计算女生人数，再计算男生人数。"
    result = react_agent.run(complex_question)
    print(f"\n复合推理响应: {result}")
    assert result is not None
    assert len(result) > 0


def test_custom_prompt(custom_agent):
    """测试自定义提示词"""
    math_question = "计算 15 × 8 + 32 ÷ 4 的结果"
    result = custom_agent.run(math_question)
    print(f"\n自定义提示词响应: {result}")
    assert result is not None
    assert len(result) > 0
