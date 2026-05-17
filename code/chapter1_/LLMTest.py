import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_ID = os.getenv("OPENAI_MODEL_ID")

prompt = "介绍一下你自己"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
response = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{'role': 'user', 'content': prompt}],
    stream=False
)
answer = response.choices[0].message.content
print(answer)

"""
你好呀！很高兴认识你！😊

我是**DeepSeek**，由**深度求索公司**创造的AI助手。让我给你介绍一下我的“特长”：

✨ **我的能力亮点**：
- **完全免费**：没错，我不收费！无论问多少问题都免费
- **超大上下文**：1M token容量，可以一次性处理像《三体》三部曲那么长的内容
- **多文件支持**：能处理图片、PDF、Word、Excel、PPT、TXT等文件，从中提取文字信息
- **联网搜索**：需要时可以帮你搜索最新信息（需要手动开启）
- **语音输入**：App端支持语音交流

📱 **使用方式**：
- 网页版和App都可以用
- App可以在官方应用商店下载

💡 **我能帮你做什么**：
- 回答问题、解释概念
- 写作、翻译、总结
- 代码编写和调试
- 数据分析
- 创意头脑风暴
- 还有很多很多...

我的知识截止到**2025年5月**，虽然不能生成图片和视频，但文字处理方面我可是很擅长的！

有什么我可以帮你的吗？无论是学习、工作还是日常问题，尽管问我！🌟
"""