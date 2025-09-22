import os
import json
import asyncio
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack

class ToolDefinition(TypedDict):
    name:str
    description: str
    parameters:dict

class MCP_Tester:
    def __init__(self) -> None:
        self.sessions: List[ClientSession] = []
        self.available_tools: List[Dict] = []
        self.tool_to_session: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            await session.initialize()
            self.sessions.append(session)

            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
            
            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description if tool.description else "",
                        "parameters": tool.inputSchema
                    }
                })

        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    tester = MCP_Tester()
    try:
        await tester.connect_to_servers()

        api_key = os.environ.get("OPENROUTER_API_KEY")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[
                {
                    "role": "user",
                    "content": "I want to fetch https://en.wikipedia.org/wiki/Japan",
                }
            ],
            tools=tester.available_tools,
        )
        print(response)

    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
