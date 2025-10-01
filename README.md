# Deep Learning MCP Course Project (Modified for OpenRouter)

This project is based on the ["Multi-Context Prompting for Rich-Context AI Applications"](https://www.deeplearning.ai/short-courses/mcp-build-rich-context-ai-apps-with-anthropic/) course from DeepLearning.AI.

The original project has been modified to use free models from [OpenRouter](https://openrouter.ai/).

## Setup

1.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Set up your OpenRouter API key as an environment variable:
    ```bash
    export OPENROUTER_API_KEY="your-api-key"
    ```

## Usage

To run the main application:

```bash
python mcp_chatbot_openrouter.py
```

This will start the chatbot, which now uses a model from OpenRouter to answer questions based on the provided context.
