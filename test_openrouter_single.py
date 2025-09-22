import os
from openai import OpenAI

# Get the API key from the environment variable
api_key = os.environ.get("OPENROUTER_API_KEY")

free_model = "x-ai/grok-4-fast:free"
# Initialize the OpenAI client with the OpenRouter API base URL and API key
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# Define the tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for papers on arXiv based on a topic and store their information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to retrieve",
                    },
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_info",
            "description": "Search for information about a specific paper across all topic directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The ID of the paper to look for",
                    },
                },
                "required": ["paper_id"],
            },
        },
    },
]

# Make the chat completion request
try:
    response = client.chat.completions.create(
        # model="openai/gpt-oss-20b:free",
        # model="moonshotai/kimi-k2:free",
        model=free_model,
        messages=[
            {
                "role": "user",
                "content": "I want to search research paper on arxiv about physics.",
            }
        ],
        tools=tools,
    )
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
