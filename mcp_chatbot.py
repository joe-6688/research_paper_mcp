import os
import json
import re
import google.generativeai as genai
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
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def clean_schema(d):
    if isinstance(d, dict):
        d.pop('title', None)
        d.pop('default', None)
        d.pop('additionalProperties', None)
        d.pop('$schema',None)
        d.pop('minLength',None)
        d.pop('exclusiveMaximum',None)
        d.pop('exclusiveMinimum',None)
        d.pop('minimum',None)
        for value in d.values():
            clean_schema(value)
    elif isinstance(d, list):
        for item in d:
            clean_schema(item)

class ToolDefinition(TypedDict):
    name:str
    description: str
    # input_schema: dict
    parameters:dict
            

def response_has_tool_call(response):
    if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call.name:
                return True
    return False

class MCP_ChatBot:

    def __init__(self) -> None:
        # Initialize session and client objects
        self.sessions: List[ClientSession] = []
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

        self.exit_stack = AsyncExitStack()
        
        self.gemini_client = genai.GenerativeModel('gemini-2.5-flash')

        # self.session: ClientSession
        # self.available_tools: List[dict] = []

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
                    "name": tool.name,
                    "description": tool.description if tool.description else "",
                    "parameters": tool.inputSchema
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
            clean_schema(self.available_tools)
            print(f"        [Debug] cleaned available_tools: {self.available_tools}")
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise
    
    async def process_query(self, query):
        messages = [
            {'role':'user',
              'parts': [{'text': query}]
             }
        ]

        response = self.gemini_client.generate_content(
            contents=messages, # type: ignore
            tools=[{"function_declarations": self.available_tools}], # type: ignore
        )
        
        print(f"[DEBUG] First response: {response}")

        message = response.candidates[0].content.parts[0]
        messages.append({'role': 'model', 'parts': response.candidates[0].content.parts})

        if response_has_tool_call(response):
            tool_calls = response.candidates[0].content.parts
            print(f"[DEBUG] Tool calls: {tool_calls}")

            for tool_call in tool_calls:
                if hasattr(tool_call, 'function_call'):
                    tool_name = tool_call.function_call.name
                    tool_args = tool_call.function_call.args
                    tool_args_dict = dict(tool_args)
                    print(f"[DEBUG] Calling tool {tool_name} with args {tool_args_dict}")

                    session = self.tool_to_session[tool_name]
                    result = await session.call_tool(tool_name,arguments=tool_args_dict) # type: ignore
                    print(f"[DEBUG] Tool result: {result}")

                    if tool_name == 'fetch' and result.content and '<error>Content truncated.' in result.content[0].text:
                        match = re.search(r'start_index of (\d+)', result.content[0].text)
                        if match:
                            start_index = int(match.group(1))
                            tool_args_dict['start_index'] = start_index
                            result = await session.call_tool(tool_name,arguments=tool_args_dict) # type: ignore
                            print(f"[DEBUG] Tool result (refetched): {result}")


                    messages.append({
                        'role': 'tool',
                        'parts': [{'function_response': {'name': tool_name, 'response': {'content': [item.text for item in result.content]}}}]
                    })
            
            pprint(f"[DEBUG] Messages before second call: {messages}")
            response = self.gemini_client.generate_content(
                contents=messages
            )
            pprint(f"[DEBUG] Second response: {response}")
            
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts and len(response.candidates[0].content.parts) > 0:
                messages.append(response.candidates[0].content)
                print(response.candidates[0].content.parts[0].text)
        else:
            if message and message.text:
                print(message.text)

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
                # print(f"\nError: {e}")
                print("\nError occurred:")
                traceback.print_exc()

    async def cleanup(self):
        await self.exit_stack.aclose()

    # async def connect_to_server_and_run(self):
    #     # Create server parameters for stdio connection
    #     server_params = StdioServerParameters(
    #         command="uv", #Executable
    #         args = ["run", "research_server.py"], #Optional arguments
    #         env=None, #Optional environment variables
    #     )

    #     async with stdio_client(server_params) as (read, write):
    #         async with ClientSession(read, write) as session:
    #             self.session = session

    #             #Initialize the connection
    #             await session.initialize()

    #             #List availabe tools
    #             response = await session.list_tools()

    #             tools = response.tools
    #             print("\nConnected to server with tools:",[tool.name for tool in tools])

    #             self.available_tools=[{
    #                 "function_declarations": [
    #                     {
    #                         "name":tool.name,
    #                         "description":tool.description,
    #                         "parameters": tool.inputSchema,
    #                     }
    #                     for tool in response.tools
    #             ]}]
    #             clean_schema(self.available_tools)

    #             await self.chat_loop()





async def main():
    chatbot = MCP_ChatBot()
    # await chatbot.connect_to_server_and_run()
    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
