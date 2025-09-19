import os
import json
from openai import OpenAI
from urllib import response
from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio



#For Jupyter Notebook
# nest_asyncio.apply()

load_dotenv()

free_model = "x-ai/grok-4-fast:free"

class MCP_ChatBot:

    def __init__(self) -> None:
        # Initialize session and client objects
        self.session: ClientSession
        self.anthropic = Anthropic()
        self.openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        self.available_tools: List[dict] = []

    
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

                result = await self.session.call_tool(tool_name,arguments=json.loads(tool_args)) # type: ignore
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

    async def connect_to_server_and_run(self):
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv", #Executable
            args = ["run", "research_server.py"], #Optional arguments
            env=None, #Optional environment variables
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session

                #Initialize the connection
                await session.initialize()

                #List availabe tools
                response = await session.list_tools()

                tools = response.tools
                print("\nConnected to server with tools:",[tool.name for tool in tools])

                self.available_tools=[
                    {
                        "type": "function",
                        "function": {
                            "name":tool.name,
                            "description":tool.description,
                            "parameters": tool.inputSchema,
                        }
                     }
                     for tool in response.tools
                ]

                await self.chat_loop()





async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
