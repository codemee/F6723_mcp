import os
import sys
import json
import asyncio
import time
import pickle
from typing import Callable
from dotenv import load_dotenv
from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from contextlib import AsyncExitStack
from google import genai
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from google_search import google_search

load_dotenv()
client = genai.Client()
async_exit_stack = AsyncExitStack()
console = Console()
hist_file = "hist.pkl"

async def get_remote_mcp_session(info:dict) -> ClientSession:
    if info.get("type", None) == "http":
        read, write, _ = (
            await async_exit_stack.enter_async_context(
                streamable_http_client(url=info["url"])
            )
        )
    elif "url" in info:
        read, write = (
            await async_exit_stack.enter_async_context(
                sse_client(url=info["url"])
            )
        )
    elif "command" in info:
        stdio_server_params = StdioServerParameters(**info)
        read, write = (
            await async_exit_stack.enter_async_context(
                stdio_client(stdio_server_params)
            )
        )
    session = await async_exit_stack.enter_async_context(
        ClientSession(read, write)
    )
    await session.initialize()
    return session

async def load_mcp():
    sessions = []

    if (not os.path.exists("mcp_servers.json") or
        not os.path.isfile("mcp_servers.json")):
        return sessions

    with open('mcp_servers.json', 'r') as f:
        mcp_servers = json.load(f)
        try:
            server_infos = tuple(
                mcp_servers['mcp_servers'].items()
            )
        except (KeyError, TypeError) as e:
            print(
                f"Error: mcp_servers.json 格式錯誤 - {e}",
                file=sys.stderr
            )
            return sessions

    for name, info in server_infos:
        print(f"啟動 MCP 伺服器 {name}...", end="")
        session = await get_remote_mcp_session(info)
        sessions.append(session)
        print(f"OK")
    return sessions

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
    
    results = []
    while True:
        if not results:
            prompt = console.input("請輸入訊息(按 ⏎ 結束): ")  
            if prompt.strip() == "":
                break
            history.append(prompt)
            contents = history + results
        else:
            contents += results
        text = ''
        async for response in await (
            client.aio.models.generate_content_stream(
                model="gemini-2.5-flash",
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    tools=tools + sessions,
                    system_instruction=(
                        f"現在 GMT 時間："
                        f"{time.strftime("%c", time.gmtime())}\n"
                        "請使用繁體中文"
                        "以 Markdown 格式回覆"
                    ),
                    automatic_function_calling=(
                        genai.types.AutomaticFunctionCallingConfig(
                            disable=True
                        )
                    ),
                )
            )
        ):
            results = await call_functions(
                response, 
                tools, sessions
            )
            if results:
                continue
            text += response.text or ""
            for hook in hooks:
                hook(response)

        if text:
            history.append({
                "role": "model",
                "parts":[{"text": text}]
            })

    if history:
        with open(hist_file, 'wb') as f:
            pickle.dump(history, f)

async def call_functions(
    response: genai.types.GenerateContentResponse,
    tools: list[Callable[[dict], str]],
    sessions: list[ClientSession],
):
    results = []

    # 不需要叫用函式
    if not response.function_calls:
        return results
    
    # 先加入原本的回應
    results.append(response.candidates[0].content)
    # 依序叫用函式
    for function_call in response.function_calls:
        name = function_call.name
        args = function_call.args
        console.log(f"叫用 {name}(**{args})", markup=False)
        result = None
        # 先檢查工具清單
        for tool in tools:
            if tool.__name__ == name:
                result = tool(**args)
                break
        # 如果沒有找到，再檢查 MCP 清單
        if result == None:
            for session in sessions:
                tool_list = await session.list_tools()
                for tool in tool_list.tools:
                    if tool.name == name:
                        result = (await session.call_tool(
                            name, 
                            args
                        )).content[0].text                    
                        break
                if not result == None:
                    break
        if not result == None:
            results.append(
                genai.types.Part.from_function_response(
                    name=name,
                    response={'result': result}
                )
            )
    return results


live: Live = None
text: str = ""

def show_text(response: genai.types.GenerateContentResponse):
    global live, text
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
    if not response.automatic_function_calling_history:
        return
    for content in response.automatic_function_calling_history:
        for part in content.parts:
            if part.function_call:
                name = part.function_call.name
                args = part.function_call.args
                console.log(
                    f"使用 {name}(**{args})", 
                    markup=False
                )
            # elif part.function_response:
            #     name = part.function_response.name
            #     response = part.function_response.response
            #     console.print(
            #         f"工具 {name} 的回應: {response}", 
            #         markup=False
            #     )

async def main():
    hooks = [show_afc, show_text]
    tools = [google_search]
    try:
        sessions = await load_mcp()
        await chat(tools, sessions, hooks)
    except KeyboardInterrupt:
        print("使用者中斷")
    finally:
        await async_exit_stack.aclose()
        print("程式結束")

asyncio.run(main())
