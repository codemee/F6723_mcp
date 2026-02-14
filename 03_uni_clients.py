import os
import sys
import json
import asyncio
import time
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

async def chat(sessions: list[ClientSession]):
    while True:
        prompt = console.input("請輸入訊息(按 ⏎ 結束): ")  
        if prompt.strip() == "":
            break
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                tools=sessions,
                system_instruction=(
                    f"現在 GMT 時間："
                    f"{time.strftime("%c", time.gmtime())}\n"
                    "請使用繁體中文"
                    "以 Markdown 格式回覆"
                )
            ),
        )
        console.print(Markdown(response.text))

async def main():
    try:
        sessions = await load_mcp()
        await chat(sessions)
    except KeyboardInterrupt:
        print("使用者中斷")
    finally:
        await async_exit_stack.aclose()
        print("程式結束")

asyncio.run(main())
