import ollama
import asyncio

async def main():
    client = ollama.AsyncClient()
    messages = [
        {'role':'user',
         'content': 'I want to search research paper on arxiv about physics.'
         }
    ]
    tools = [
        {
            'name': 'search_papers',
            'description': 'Search for papers on arXiv based on a topic and store their information.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'topic': {'type': 'string'},
                    'max_results': {'type': 'integer', 'default': 5}
                },
                'required': ['topic']
            }
        },
        {
            'name': 'extract_info',
            'description': 'Search for information about a specific paper across all topic directories.',
            'input_schema': {
                'type': 'object',
                'properties': {
                    'paper_id': {'type': 'string'}
                },
                'required': ['paper_id']
            }
        }
    ]
    try:
        response = await client.chat(
            model='qwen2:7b',
            messages=messages,
            tools=tools,
        )
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
