import asyncio
from google import genai
from mcp.client.sse import sse_client
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
    async with sse_client(url="http://localhost:3001/sse") as (
        read, write
    ):
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
                    # 若要停用 function calling 自動叫用，要如下指定
                    # 但就要自己根據回覆內容手動透過 session 去叫用工具
                    # 請參考 F5762
                    # automatic_function_calling=(
                    #     genai.types.AutomaticFunctionCallingConfig(
                    #         disable=True
                    #     )
                    # ),
                ),
            )
            # 通常都會生成 Markdown 內容
            console.print(Markdown(response.text))

asyncio.run(run_sse())
