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
        self.session_list: List[ClientSession] = []
        # self.available_tools: List[Dict] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

        # Tools, Resource and Prompts to session
        self.sessions:Dict = {}

        self.available_tools: List[Dict] = []
        self.available_prompts: List[Dict] = []

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

                # session = self.tool_to_session[tool_name]
                session = self.sessions[tool_name]
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

    async def get_resource(self, resource_uri):
        session = self.sessions.get(resource_uri)

        # Fallback for papers URIs - try any papers resource session, the topic user gave may not exists.
        if not session and resource_uri.startswith("papers://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("papers://"):
                    session = sess
                    break
            
        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return
        
        try:
            result = await session.read_resource(uri = resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
        return ""
    
    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return
        
        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    print(f"    - {arg_name}")

    async def execute_prompts(self, prompt_name, args):
        """Execute a prompt with the given arguments."""        
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return
        
        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content
                
                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(item.text if hasattr(item, 'text') else str(item) 
                                  for item in prompt_content)
                
                print(f"\nExecuting prompt '{prompt_name}'...")
                print(f"    [Debug] text sent to query: {text}")
                await self.process_query(text)
        except Exception as e:
            print(f"Error {e}")
            traceback.print_exc()

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break
                if not query:
                    continue
                
                if query.startswith('@'):      #Check for @resource syntax first
                    #Remove @sign
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
            
                elif query.startswith('/'):    #Check for prompt syntax
                    parts = query.split()
                    command= parts[0].lower()
                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: prompt <name> <arg1=value1> <arg2=value2>")
                        prompt_name = parts[1]
                        args = {}
                        #parse arguments
                        for arg in parts[2:]:
                            if '=' in arg:
                                key,value = arg.split('=',1)
                                args[key] = value
                        await self.execute_prompts(prompt_name, args)

                else: # Process the query by calling LLM
                    await self.process_query(query)
                    print("\n")
            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()

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
            self.session_list.append(session)

            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
            
            for tool in tools: # new
                # self.tool_to_session[tool.name] = session
                self.sessions[tool.name] = session
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description if tool.description else "",
                        "parameters": tool.inputSchema
                    }
                })

            # List avaialbe resources
            resource_response = await session.list_resources()
            if resource_response and resource_response.resources:
                for res in resource_response.resources:
                    resource_uri = str(res.uri)
                    self.sessions[resource_uri] = session

            # List avaialbe prompts
            prompts_response = await session.list_prompts()
            if prompts_response and prompts_response.prompts:
                for prompt in prompts_response.prompts:
                    self.sessions[prompt.name] = session
                    self.available_prompts.append(
                        {
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        }
                    )
            

        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")
            traceback.print_exc()

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
