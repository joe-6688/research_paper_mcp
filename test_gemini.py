import os
import google.generativeai as genai

# Get the API key from the environment variable
api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Initialize the GenerativeModel
client = genai.GenerativeModel('gemini-2.0-flash')

# Define the tools
tools = [
    {
        "function_declarations": [
            {
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
            {
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
        ]
    }
]

# Make the chat completion request
try:
    response = client.generate_content(
        contents=[
            {
                "role": "user",
                "parts": [{"text": "I want to search research paper on arxiv about physics."}]
            }
        ],
        tools=tools,
    )
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")
