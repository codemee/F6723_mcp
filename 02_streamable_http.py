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
    async with streamable_http_client(
        url="http://localhost:8080/mcp"
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            # 連接 MCP 服器
            await session.initialize()

            prompt = f"WBC 中華隊備戰狀況？"
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    tools=[session],
                ),
            )
            console.print(Markdown(response.text))

asyncio.run(run_sse())