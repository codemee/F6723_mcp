import asyncio
from google import genai
from mcp.client.stdio import stdio_client
from mcp import ClientSession
from mcp import StdioServerParameters
from rich.console import Console
from rich.markdown import Markdown
import dotenv
import os

dotenv.load_dotenv()
console = Console()
client = genai.Client()


def get_desktop_path():
    """
    取得當前使用者的桌面路徑（跨平台）。

    依序檢查常見的桌面資料夾名稱（Desktop、桌面），
    若皆不存在則回傳使用者家目錄。

    Returns:
        str: 桌面路徑或家目錄路徑
    """
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
    stdio_server_params = StdioServerParameters(
        command="npx",
        args=[
            "-y", 
            "@modelcontextprotocol/server-filesystem", 
            get_desktop_path()
        ],
    )
    async with stdio_client(stdio_server_params) as (
        read, write
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()

            prompt = f"我的桌面上（不含子資料夾）有哪些圖檔？"
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    tools=[get_desktop_path, session],
                ),
            )
            console.print(Markdown(response.text))


asyncio.run(run_stdio())