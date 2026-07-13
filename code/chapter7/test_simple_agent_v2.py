# test_simple_agent.py
import pytest
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ToolRegistry
from hello_agents.tools import CalculatorTool
from my_simple_agent import MySimpleAgent

# 加载环境变量
load_dotenv()

# @pytest.fixture(scope="function")  # 默认：每个测试函数都创建新实例
# @pytest.fixture(scope="class")     # 每个测试类创建一次
# @pytest.fixture(scope="module")    # 整个模块共享一个实例
# @pytest.fixture(scope="session")   # 整个测试会话共享一个实例
@pytest.fixture(scope="module")
def llm():
    return HelloAgentsLLM()


@pytest.fixture
def basic_agent(llm):
    return MySimpleAgent(
        name="基础助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。"
    )


@pytest.fixture
def enhanced_agent(llm):
    tool_registry = ToolRegistry()
    calculator = CalculatorTool()
    tool_registry.register_tool(calculator)
    return MySimpleAgent(
        name="增强助手",
        llm=llm,
        system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
        tool_registry=tool_registry,
        enable_tool_calling=True
    )

# pytest 的自动发现规则：
# 文件名匹配：test_*.py 或 *_test.py
# 函数名匹配：以 test_ 开头的函数
# 自动收集：pytest 扫描到这些函数后自动执行
def test_basic_conversation(basic_agent):
    response = basic_agent.run("你好，请介绍一下自己")
    print(f"\n基础对话响应: {response}")
    assert response is not None
    assert len(response) > 0


def test_tool_enhanced_conversation(enhanced_agent):
    response = enhanced_agent.run("请帮我计算 15 * 8 + 32")
    print(f"\n工具增强响应: {response}")
    assert response is not None
    assert "152" in response or "计算" in response


def test_stream_response(basic_agent):
    chunks = list(basic_agent.stream_run("请解释什么是人工智能"))
    print(f"\n流式响应块数: {len(chunks)}")
    assert len(chunks) > 0


def test_dynamic_tool_management(basic_agent):
    calculator = CalculatorTool()

    assert not basic_agent.has_tools()
    basic_agent.add_tool(calculator)
    assert basic_agent.has_tools()
    assert "calculator" in basic_agent.list_tools()
    print(f"\n可用工具: {basic_agent.list_tools()}")
