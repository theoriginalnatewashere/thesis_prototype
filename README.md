# Data Pod Assistant with RAG

A Streamlit application that allows users to chat with their personal data using Retrieval Augmented Generation (RAG). The app supports multiple LLM providers (OpenAI and Groq) and includes web search capabilities through Tavily.

## Features

- 📁 File Upload: Support for TXT, MD, and CSV files
- 🤖 Multiple LLM Providers: OpenAI (GPT-3.5/4) and Groq (Llama3/Mixtral)
- 🔍 Web Search Integration: Tavily search for up-to-date information
- 💾 Session Management: Chat history and interaction logging
- 📊 Analytics: Track response times and interaction statistics
- 🔒 Privacy-First: Local processing, no data storage

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API keys in Streamlit secrets:
   ```toml
   # .streamlit/secrets.toml
   OPENAI_API_KEY = "your-openai-key"
   GROQ_API_KEY = "your-groq-key"
   TAVILY_API_KEY = "your-tavily-key"
   ```

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Add your API keys in the Streamlit Cloud secrets management
5. Deploy!

## Local Development

Run the app locally:
```bash
streamlit run streamlit.py
```

## Environment Variables

The following API keys are required:
- `OPENAI_API_KEY` (for OpenAI models)
- `GROQ_API_KEY` (for Groq models)
- `TAVILY_API_KEY` (for web search)

## Usage

1. Select your preferred LLM provider
2. Enter the required API keys
3. Upload your data file
4. Start chatting with your data!

## License

MIT License 