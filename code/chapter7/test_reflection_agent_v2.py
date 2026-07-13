# test_reflection_agent_v2.py
import pytest
from dotenv import load_dotenv
from hello_agents import HelloAgentsLLM, ReflectionAgent

# 加载环境变量
load_dotenv()


@pytest.fixture(scope="module")
def llm():
    return HelloAgentsLLM()


@pytest.fixture
def general_agent(llm):
    """使用默认通用提示词的反思Agent"""
    return ReflectionAgent(name="我的反思助手", llm=llm)


@pytest.fixture
def code_agent(llm):
    """使用自定义代码生成提示词的反思Agent"""
    code_prompts = {
        "initial": "你是Python专家，请编写函数：{task}",
        "reflect": "请审查代码的算法效率：\n任务：{task}\n代码：{content}",
        "refine": "请根据反馈优化代码：\n任务：{task}\n反馈：{feedback}"
    }
    return ReflectionAgent(
        name="我的代码生成助手",
        llm=llm,
        custom_prompts=code_prompts
    )


def test_general_reflection(general_agent):
    """测试默认通用提示词反思"""
    result = general_agent.run("写一篇关于人工智能发展历程的简短文章")
    print(f"\n通用反思响应: {result}")
    assert result is not None
    assert len(result) > 0


def test_code_reflection(code_agent):
    """测试自定义代码生成反思"""
    result = code_agent.run("计算斐波那契数列的第n项")
    print(f"\n代码反思响应: {result}")
    assert result is not None
    assert len(result) > 0
