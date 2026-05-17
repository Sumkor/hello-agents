import json
import os
import re

import requests
from dotenv import load_dotenv
from openai import OpenAI

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 行动格式:
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动，每次回复只输出一对Thought-Action：
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 任务完成:
当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `finish(answer="...")` 来输出最终答案。

请开始吧！
"""


class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端。
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("正在调用大语言模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误：调用语言模型服务时出错。"


def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status()
        # 解析返回的JSON数据
        data = response.json()

        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        # 格式化成自然语言返回
        return f"{city}当前天气：{weather_desc}，气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误：查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用LangSearch API搜索并返回优化后的景点推荐。
    """
    api_key = os.getenv("LANGSEARCH_API_KEY")
    if not api_key:
        return "错误：未配置 LANGSEARCH_API_KEY。"

    url = "https://api.langsearch.com/v1/web-search"
    query = f"{city}在{weather}天气下最值得去的旅游景点推荐及理由"

    payload = json.dumps(
        {
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": 5,
        }
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        web_pages = data.get("data", {}).get("webPages", {})
        results = web_pages.get("value", [])

        if not results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        first_result = results[0]
        summary = first_result.get("summary") or first_result.get("snippet")
        title = first_result.get("name", "未知景点")
        source_url = first_result.get("url", "")

        if summary:
            return f"推荐景点：{title}\n推荐理由：{summary}\n参考链接：{source_url}"

        formatted_results = []
        for result in results:
            name = result.get("name", "未知景点")
            snippet = result.get("summary") or result.get("snippet") or "暂无简介"
            formatted_results.append(f"- {name}: {snippet}")

        return "根据搜索，为您找到以下景点推荐：\n" + "\n".join(formatted_results)

    except requests.exceptions.RequestException as e:
        return f"错误：执行 LangSearch 搜索时遇到网络问题 - {e}"
    except (ValueError, KeyError, IndexError) as e:
        return f"错误：解析 LangSearch 搜索结果失败 - {e}"


# --- 1. 配置LLM客户端 ---
# 请根据您使用的服务，将这里替换成对应的凭证和地址
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_ID = os.getenv("OPENAI_MODEL_ID")

llm = OpenAICompatibleClient(model=MODEL_ID, api_key=API_KEY, base_url=BASE_URL)

# 将所有工具函数放入一个字典，方便后续调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}

# --- 2. 初始化 ---
user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
prompt_history = [f"用户请求: {user_prompt}"]

print(f"用户输入: {user_prompt}\n" + "=" * 40)

# --- 3. 运行主循环 ---
for i in range(5):  # 设置最大循环次数
    print(f"--- 循环 {i + 1} ---\n")

    # 3.1. 构建Prompt
    full_prompt = "\n".join(prompt_history)

    # 3.2. 调用LLM进行思考
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    # 模型可能会输出多余的Thought-Action，需要截断
    match = re.search(
        r"(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)",
        llm_output,
        re.DOTALL,
    )
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            print("已截断多余的 Thought-Action 对")
    print(f"模型输出:\n{llm_output}\n")
    prompt_history.append(llm_output)

    # 3.3. 解析并执行行动
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        print("解析错误：模型输出中未找到 Action。")
        break
    action_str = action_match.group(1).strip()

    if action_str.startswith("finish"):
        finish_match = re.search(
            r'finish\(\s*answer\s*=\s*["\'](.*?)["\']\s*\)',
            action_str,
            re.DOTALL,
        )
        if not finish_match:
            print(
                f"解析错误：无法解析 finish 的 answer 参数。action_str={action_str!r}"
            )
            break

        final_answer = finish_match.group(1)
        print(f"任务完成，最终答案: {final_answer}")
        break

    tool_match = re.search(r"(\w+)\(", action_str)
    args_match = re.search(r"\((.*)\)", action_str, re.DOTALL)
    if not tool_match or not args_match:
        print(f"解析错误：无法解析工具调用。action_str={action_str!r}")
        break

    tool_name = tool_match.group(1)
    args_str = args_match.group(1)
    kwargs = dict(re.findall(r'(\w+)\s*=\s*["\']([^"\']*)["\']', args_str, re.DOTALL))

    if tool_name in available_tools:
        observation = available_tools[tool_name](**kwargs)
    else:
        observation = f"错误：未定义的工具 '{tool_name}'"

    # 3.4. 记录观察结果
    observation_str = f"Observation: {observation}"
    print(f"{observation_str}\n" + "=" * 40)
    prompt_history.append(observation_str)

"""
用户输入: 你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。
========================================
--- 循环 1 ---

正在调用大语言模型...
大语言模型响应成功。
模型输出:
Thought: 用户想查询北京的天气，然后根据天气推荐旅游景点。我需要先调用 get_weather 工具获取北京的天气信息。

Action: get_weather(city="北京")

Observation: 北京当前天气：Light Rain, Mist，气温18摄氏度
========================================
--- 循环 2 ---

正在调用大语言模型...
大语言模型响应成功。
已截断多余的 Thought-Action 对
模型输出:
Thought: 从天气查询结果得知，北京当前是 Light Rain, Mist（小雨有雾），气温18摄氏度。接下来我需要根据这个天气信息来搜索推荐的旅游景点。

Action: get_attraction(city="北京", weather="Light Rain, Mist")

Observation: 推荐景点：北京旅游攻略,不可辜负的好天气,虽然不能到处旅游,但可以在胡同中--穷游网
推荐理由：不可辜负的好天气,虽然不能到处旅游,但可以在胡同中感受到地道的越南风味[太阳][太阳]周边游去哪儿北京
========================================
--- 循环 3 ---

正在调用大语言模型...
大语言模型响应成功。
模型输出:
Thought: 已获得天气信息和推荐景点。根据结果，虽然是小雨有雾的天气，但仍然推荐了适合的景点——胡同游。我现在可以整理信息回答用户了。

Action: finish(answer="今天北京天气是小雨有雾，气温18摄氏度。虽然天气不太晴朗，但推荐您去北京的胡同里逛逛，在朦胧的雨中感受老北京的韵味和地道的美食，是个不错的雨天选择。")

任务完成，最终答案: 今天北京天气是小雨有雾，气温18摄氏度。虽然天气不太晴朗，但推荐您去北京的胡同里逛逛，在朦胧的雨中感受老北京的韵味和地道的美食，是个不错的雨天选择。

"""
