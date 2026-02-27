import os
import asyncio
import time
import pickle
from typing import Callable
from dotenv import load_dotenv
from mcp import ClientSession
from google import genai
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from mcp_utils import load_mcp, close_mcp
from google_search import google_search

load_dotenv()
client = genai.Client()
console = Console()
hist_file = "hist.pkl"

async def chat(
    tools: list,
    sessions: list[ClientSession], 
    hooks: list[
        Callable[[genai.types.GenerateContentResponse], None]
    ]
):
    if os.path.exists(hist_file):
        with open(hist_file, 'rb') as f:
            history = pickle.load(f)
    else:
        history = []
    while True:
        prompt = console.input("請輸入訊息(按 ⏎ 結束): ")  
        if prompt.strip() == "":
            break
        history.append(prompt)
        async for response in await (
            client.aio.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=history,
                config=genai.types.GenerateContentConfig(
                    tools=tools + sessions,
                    system_instruction=(
                        f"現在 GMT 時間："
                        f"{time.strftime("%c", time.gmtime())}\n"
                        "請使用繁體中文"
                        "以 Markdown 格式回覆"
                    )
                )
            )
        ):
            for hook in hooks:
                hook(response)
            if not response.function_calls:
                history.append(response.candidates[0].content)


    if history:
        with open(hist_file, 'wb') as f:
            pickle.dump(history, f)

live: Live | None = None
text: str = ""

def show_text(response: genai.types.GenerateContentResponse):
    global live, text
    if response.function_calls:
        return
    if not live:
        live = Live(
            Markdown(""),
            console=console,
            refresh_per_second=10,
        )
        live.start()
    text += response.text or ""
    live.update(Markdown(text))
    candidates = response.candidates or []
    if (
        candidates[0].finish_reason == 
        genai.types.FinishReason.STOP
    ):
        live.stop()
        live = None
        text = ""

def show_afc(response: genai.types.GenerateContentResponse):
    candidates = response.candidates
    if (
        not candidates[0].finish_reason == 
        genai.types.FinishReason.STOP
    ):
        return
    if not response.automatic_function_calling_history:
        return
    content = response.automatic_function_calling_history[-2]
    for part in content.parts:
        if part.function_call:
            name = part.function_call.name
            args = part.function_call.args
            console.print(f" →{name}(**{args})")

async def main():
    hooks = [show_afc, show_text]
    tools = [google_search]
    try:
        sessions = await load_mcp()
        await chat(tools, sessions, hooks)
    except Exception as e:
        console.print(f"[red]錯誤: {e}[/red]")
    finally:
        await close_mcp()
        console.print("程式結束")

asyncio.run(main())
