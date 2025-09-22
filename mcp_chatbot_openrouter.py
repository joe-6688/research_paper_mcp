import os
import json
from openai import OpenAI
from urllib import response
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List,Dict,TypedDict
from contextlib import AsyncExitStack
import asyncio
import nest_asyncio
import traceback
from pprint import pprint



#For Jupyter Notebook
# nest_asyncio.apply()

load_dotenv()

free_model = "x-ai/grok-4-fast:free"
# free_model = "deepseek/deepseek-chat-v3.1:free"

class ToolDefinition(TypedDict):
    name:str
    description: str
    parameters:dict

class MCP_ChatBot:

    def __init__(self) -> None:
        # Initialize session and client objects
        self.sessions: List[ClientSession] = []
        self.available_tools: List[Dict] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

        self.exit_stack = AsyncExitStack()
        
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )

    
    async def process_query(self, query):
        messages = [
            {'role':'user',
              'content': query
             }
        ]

        response = self.openai_client.chat.completions.create(
            model=free_model,
            messages=messages, # type: ignore
            tools=self.available_tools, # type: ignore
        )
        
        print(f"[DEBUG] First response: {response}")

        message = response.choices[0].message
        messages.append(message)

        if message.tool_calls:
            tool_calls = message.tool_calls
            print(f"[DEBUG] Tool calls: {tool_calls}")

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                print(f"[DEBUG] Calling tool {tool_name} with args {tool_args}")

                session = self.tool_to_session[tool_name]
                result = await session.call_tool(tool_name,arguments=json.loads(tool_args)) # type: ignore
                print(f"[DEBUG] Tool result: {result}")

                messages.append({
                    'tool_call_id': tool_call.id,
                    'role': 'tool',
                    'name': tool_name,
                    'content': result.content
                })
            
            print(f"[DEBUG] Messages before second call: {messages}")
            response = self.openai_client.chat.completions.create(
                model=free_model,
                messages=messages
            )
            print(f"[DEBUG] Second response: {response}")
            
            print(response.choices[0].message.content)
        else:
            print(response.choices[0].message.content)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                if query:
                    await self.process_query(query)
                    print("\n")
            except Exception as e:
                print(f"\nError: {e}")

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ) # new 

            await session.initialize()
            self.sessions.append(session)

                        # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
            
            for tool in tools: # new
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

    async def connect_to_servers(self): # new
        """Connect to all configured MCP servers."""
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
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
