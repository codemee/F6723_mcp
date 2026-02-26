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
    async with sse_client(
        url="https://gitmcp.io/openclaw/openclaw"
    ) as (
        read, write
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()

            prompt = f"OpenClaw 是用什麼開發的？"
            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    tools=[session],
                ),
            )
            console.print(Markdown(response.text))

asyncio.run(run_sse())