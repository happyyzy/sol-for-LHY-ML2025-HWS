# ask_qwen_agent.py
from qwen_agent.agents import Assistant
import time
import sys
import os
os.environ["SERPER_API_KEY"] = "727a2e31a96bb254a6abb3da59fe04fd0ac47273"

# ---------- LLM / Agent 配置（按需修改） ----------
llm_cfg = {
    'model': 'Qwen3-4B',
    'model_server': 'http://localhost:8000/v1',  # api_base
    'api_key': 'EMPTY',
}

from qwen_agent.tools import WebSearch, WebExtractor, KeywordSearch

tools = [
    WebSearch(),       # 网页搜索
    WebExtractor(),    # 网页内容抓取
    KeywordSearch(),   # 关键词搜索
    # 你也可以加其他工具，比如 CodeInterpreter, PythonExecutor 等
]


bot = Assistant(llm=llm_cfg, function_list=tools)

# ---------- 问题列表（逐条） ----------
questions = [
    "「短暫交會的旅程就此分岔」是哪個歌唱團體歌曲中的歌詞？",
]

def ask_and_get_final(bot, user_text):
    last = None
    for chunk in bot.run(messages=[{'role': 'user', 'content': user_text}]):
        if isinstance(chunk, list) and chunk:
            last_msg = chunk[-1]  # 取最新的一条消息
            if last_msg.role == "assistant" and last_msg.content:
                last = last_msg.content
    return last if last else "[NO FINAL ANSWER]"


# ---------- 主流程：遍历问题，收集答案写入 TXT ----------
output_filename = "answers.txt"
from tqdm import tqdm
import time

answers = []

# 使用 tqdm 包装 questions，显示进度条
for q in tqdm(questions, desc="Processing questions", ncols=100):
    ans = ask_and_get_final(bot, q)
    if isinstance(ans, str):
        ans_line = " ".join(ans.strip().splitlines())
    else:
        ans_line = str(ans)
    answers.append(ans_line)
    time.sleep(0.4)  # 可选，避免请求过快


# 写入文件（每行一条答案）
with open(output_filename, "w", encoding="utf-8") as f:
    for ans in answers:
        f.write(ans + "\n")

print(f"\nAll done. {len(questions)} answers saved to '{output_filename}'.")

