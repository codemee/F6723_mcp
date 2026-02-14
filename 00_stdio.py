import asyncio
from google import genai
from mcp.client.stdio import stdio_client
from mcp import ClientSession
from mcp import StdioServerParameters
from rich.console import Console
from rich.markdown import Markdown
import dotenv

dotenv.load_dotenv()
console = Console()
client = genai.Client()
import os

# 取得當前使用者的桌面路徑 (跨平台)
def get_desktop_path():
    home = os.path.expanduser("~")
    # 常見平台的桌面名稱
    possible_names = ["Desktop", "桌面"]
    for name in possible_names:
        path = os.path.join(home, name)
        if os.path.isdir(path):
            return path
    # fallback: 首選 home
    return home

async def run_stdio():
    # 建立連接 MCP 伺服器的用戶端
    # 根據 MCP 伺服器執行的訊息取得位址

    stdio_server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            get_desktop_path()
        ]
    )
    async with stdio_client(stdio_server_params) as (
        read, write
    ):
        async with ClientSession(read, write) as session:
            # 連接 MCP 伺服器
            await session.initialize()

            # 利用 MCP 伺服器提供的環境變數工具取得 PATH 變數內容
            prompt = f"我的桌面上有哪些圖檔？"
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],
                    # 如下指定可停用自動叫用
                    # automatic_function_calling=(
                    #     genai.types.AutomaticFunctionCallingConfig(
                    #         disable=True
                    #     )
                    # ),
                    # 必須根據回覆內容手動透過 session 去叫用工具
                ),
            )
            # 通常都會生成 Markdown 內容
            console.print(Markdown(response.text))

asyncio.run(run_stdio())
