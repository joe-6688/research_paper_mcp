import os
import json
import google.generativeai as genai
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
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def clean_schema(d):
    if isinstance(d, dict):
        d.pop('title', None)
        d.pop('default', None)
        for value in d.values():
            clean_schema(value)
    elif isinstance(d, list):
        for item in d:
            clean_schema(item)

class MCP_ChatBot:

    def __init__(self) -> None:
        # Initialize session and client objects
        self.session: ClientSession
        
        # self.gemini_client = genai.GenerativeModel('gemini-1.5-pro-latest')
        self.gemini_client = genai.GenerativeModel('gemini-2.0-flash')
        self.available_tools: List[dict] = []

    
    async def process_query(self, query):
        messages = [
            {'role':'user',
              'parts': [{'text': query}]
             }
        ]

        response = self.gemini_client.generate_content(
            contents=messages, # type: ignore
            tools=self.available_tools, # type: ignore
        )
        
        print(f"[DEBUG] First response: {response}")

        message = response.candidates[0].content.parts[0]
        messages.append({'role': 'model', 'parts': response.candidates[0].content.parts})

        if hasattr(message, 'function_call') and message.function_call.name != '':
            tool_calls = response.candidates[0].content.parts
            print(f"[DEBUG] Tool calls: {tool_calls}")

            for tool_call in tool_calls:
                if hasattr(tool_call, 'function_call'):
                    tool_name = tool_call.function_call.name
                    tool_args = tool_call.function_call.args
                    print(f"[DEBUG] Calling tool {tool_name} with args {tool_args}")

                    result = await self.session.call_tool(tool_name,arguments=tool_args) # type: ignore
                    print(f"[DEBUG] Tool result: {result}")

                    messages.append({
                        'role': 'tool',
                        'parts': [{'function_response': {'name': tool_name, 'response': {'content': [item.text for item in result.content]}}}]
                    })
            
            print(f"[DEBUG] Messages before second call: {messages}")
            response = self.gemini_client.generate_content(
                contents=messages
            )
            print(f"[DEBUG] Second response: {response}")
            
            print(response.candidates[0].content.parts[0].text)
        else:
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

                self.available_tools=[{
                    "function_declarations": [
                        {
                            "name":tool.name,
                            "description":tool.description,
                            "parameters": tool.inputSchema,
                        }
                        for tool in response.tools
                ]}]
                clean_schema(self.available_tools)

                await self.chat_loop()





async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()


if __name__ == "__main__":
    asyncio.run(main())
