import os
import sys
import json
import asyncio
import time
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

load_dotenv()
client = genai.Client()
async_exit_stack = AsyncExitStack()
console = Console()

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
    sessions: list[ClientSession], 
    hooks: list[
        Callable[[genai.types.GenerateContentResponse], None]
    ]
):
    chat = client.aio.chats.create(
        model="gemini-3-pro-preview",
        config=genai.types.GenerateContentConfig(
            tools=sessions,
            system_instruction=(
                f"現在 GMT 時間："
                f"{time.strftime("%c", time.gmtime())}\n"
                "請使用繁體中文"
                "以 Markdown 格式回覆"
            )
        )
    )
    while True:
        prompt = console.input("請輸入訊息(按 ⏎ 結束): ")  
        if prompt.strip() == "":
            break
        response = await chat.send_message(prompt)
        for hook in hooks:
            hook(response)

def show_text(response: genai.types.GenerateContentResponse):
    console.print(Markdown(response.text))

def show_afc(response: genai.types.GenerateContentResponse):
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
    try:
        sessions = await load_mcp()
        await chat(sessions, hooks)
    except KeyboardInterrupt:
        print("使用者中斷")
    finally:
        await async_exit_stack.aclose()
        print("程式結束")

asyncio.run(main())
