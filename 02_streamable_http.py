import asyncio
from google import genai
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession
from rich.console import Console
from rich.markdown import Markdown
import dotenv

dotenv.load_dotenv()
console = Console()
client = genai.Client()

async def run_sse():
    # 建立連接 MCP 伺服器的用戶端
    # 根據 MCP 伺服器執行的訊息取得位址
    async with streamable_http_client(
        url="http://localhost:3002/mcp"
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            # 連接 MCP 伺服器
            await session.initialize()

            # 利用 MCP 伺服器提供的環境變數工具取得 PATH 變數內容
            prompt = f"我的 PATH 包含哪些路徑？"
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],
                ),
            )
            # 通常都會生成 Markdown 內容
            console.print(Markdown(response.text))

asyncio.run(run_sse())
