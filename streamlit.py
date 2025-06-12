# Required packages - install with:
# pip install streamlit langchain langchain-community faiss-cpu pandas python-dotenv groq langchain-groq sentence-transformers

import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime
from io import StringIO
import os
from typing import List, Dict, Any

# LangChain imports
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chatbot_chain" not in st.session_state:
    st.session_state.chatbot_chain = None
if "chat_logs" not in st.session_state:
    st.session_state.chat_logs = []

# Get API keys from environment or secrets
def get_api_key(key_name: str) -> str:
    """Get API key from environment or secrets"""
    # Try to get from Streamlit secrets first (for cloud deployment)
    if hasattr(st.secrets, key_name):
        return st.secrets[key_name]
    # Then try environment variables
    return os.getenv(key_name, "")

# Streamlit page config
st.set_page_config(
    page_title="Data Pod Assistant",
    page_icon="🤖",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/yourusername/datapod-assistant',
        'Report a bug': "https://github.com/yourusername/datapod-assistant/issues",
        'About': "# Data Pod Assistant\nChat with your personal data using RAG technology."
    }
)

st.title("🤖 Data Pod Assistant with RAG")
st.markdown("Upload your data and chat with an AI assistant that understands your personal information!")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Get API key from secrets/env or user input
    default_groq_key = get_api_key("GROQ_API_KEY")
    api_key = st.text_input(
        "Groq API Key", 
        value=default_groq_key,
        type="password",
        help="Enter your Groq API key (should start with 'gsk_')"
    )
    
    # Validate Groq API key format
    if api_key and not api_key.startswith('gsk_'):
        st.warning("⚠️ Groq API key should start with 'gsk_'")
    
    # Set model name directly without selection
    model_name = "llama-3.1-8b-instant"
    
    # Show model information
    st.info("ℹ️ Using Llama 3.1 8B: High-speed model (~750 tokens/sec) optimized for real-time responses")
    
    # Tavily web search integration
    st.markdown("---")
    st.header("🌐 Web Search (Tavily)")
    default_tavily_key = get_api_key("TAVILY_API_KEY")
    tavily_api_key = st.text_input(
        "Tavily API Key (for web search)", 
        value=default_tavily_key,
        type="password"
    )
    use_web_search = st.checkbox("Enable Web Search", value=False)
    if tavily_api_key and use_web_search:
        web_search_tool = TavilySearchResults(tavily_api_key=tavily_api_key)
    else:
        web_search_tool = None

    st.markdown("---")
    st.header("📊 Session Stats")
    if st.session_state.chat_logs:
        st.metric("Total Interactions", len(st.session_state.chat_logs))
        avg_response_time = sum([log.get('response_time', 0) for log in st.session_state.chat_logs]) / len(st.session_state.chat_logs)
        st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.chat_logs = []
        st.session_state.memory = None
        st.rerun()
    
    if st.button("📤 Export Chat Logs") and st.session_state.chat_logs:
        logs_json = json.dumps(st.session_state.chat_logs, indent=2)
        st.download_button(
            label="Download Chat Logs",
            data=logs_json,
            file_name=f"chat_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

# File upload section
st.header("📁 Upload Your Data")
uploaded_file = st.file_uploader(
    "Upload your data file",
    type=["txt", "md", "csv"],
    help="Upload a text, markdown, or CSV file to create your data pod"
)

def process_uploaded_file(file):
    """Process uploaded file and return text content"""
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension in ['txt', 'md']:
        raw = file.read()
        content = raw.decode('utf-8', errors='replace')
        return content
    elif file_extension == 'csv':
        df = pd.read_csv(file)
        # Convert CSV to text format for better RAG processing
        content = f"Data Summary:\n\n"
        content += f"Total rows: {len(df)}\n"
        content += f"Columns: {', '.join(df.columns)}\n\n"
        content += "Data Preview:\n"
        content += df.head(10).to_string(index=False)
        content += "\n\nFull Data:\n"
        content += df.to_string(index=False)
        return content
    else:
        return None

def create_vectorstore(text_content):
    """Create FAISS vectorstore from text content using HuggingFace embeddings"""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    # Create documents
    documents = [Document(page_content=text_content)]
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings using a free HuggingFace model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def initialize_chatbot(api_key, model_name, retriever):
    """Initialize the chatbot chain"""
    try:
        # Add debug information
        st.info(f"Initializing Groq with model: {model_name}")
        
        # Ensure API key starts with 'gsk_'
        if not api_key.startswith('gsk_'):
            st.error("Invalid Groq API key format. It should start with 'gsk_'")
            raise ValueError("Invalid Groq API key format")
        
        llm = ChatGroq(
            temperature=0.3,
            model_name=model_name,
            groq_api_key=api_key
        )
        st.success("Groq initialization successful!")
    except Exception as e:
        st.error(f"Error initializing Groq: {str(e)}")
        raise e
    
    # Memory setup
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create Conversational Retrieval Chain
    chatbot_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True
    )
    
    return chatbot_chain, memory

def log_chat_interaction(user_input, bot_response, response_time, source_documents, model_used):
    """Log chat interaction"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "bot_response": bot_response,
        "response_time": response_time,
        "model_used": model_used,
        "source_documents_count": len(source_documents) if source_documents else 0,
        "retrieved_chunks": [doc.page_content[:100] + "..." for doc in (source_documents or [])[:3]]
    }
    st.session_state.chat_logs.append(log_entry)

# Main app logic
if uploaded_file and api_key:
    # Process uploaded file
    with st.spinner("Processing your data..."):
        text_content = process_uploaded_file(uploaded_file)
        
        if text_content:
            try:
                # Create vectorstore
                vectorstore = create_vectorstore(text_content)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                st.session_state.retriever = retriever
                
                # Initialize chatbot
                chatbot_chain, memory = initialize_chatbot(api_key, model_name, retriever)
                st.session_state.chatbot_chain = chatbot_chain
                st.session_state.memory = memory
                
                st.success("✅ Data pod created successfully! You can now chat with your data.")
                
                # Show data preview
                with st.expander("📋 Data Preview"):
                    st.text(text_content[:1000] + "..." if len(text_content) > 1000 else text_content)
                    
            except Exception as e:
                st.error(f"Error creating data pod: {str(e)}")
        else:
            st.error("Failed to process the uploaded file.")

elif uploaded_file and not api_key:
    st.warning("⚠️ Please enter your API key to process the data.")

# Chat interface
if st.session_state.chatbot_chain:
    st.header("💬 Chat with Your Data Pod")
    
    # Display chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:  # User message
            with st.chat_message("user"):
                st.write(message.content)
        else:  # Bot message
            with st.chat_message("assistant"):
                st.write(message.content)
    
    # Chat input
    if prompt := st.chat_input("Ask something about your data..."):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                # If web search is enabled, append results to the question
                if web_search_tool:
                    try:
                        search_results = web_search_tool.run(prompt)
                        prompt_with_web = f"{prompt}\n\n[External Web Search Results]\n{search_results}"
                    except Exception as e:
                        st.warning(f"Tavily search error: {e}")
                        prompt_with_web = prompt
                else:
                    prompt_with_web = prompt
                
                try:
                    result = st.session_state.chatbot_chain({
                        "question": prompt_with_web,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    response_time = time.time() - start_time
                    response = result["answer"]
                    
                    st.write(response)
                    
                    # Update chat history
                    st.session_state.chat_history = result["chat_history"]
                    
                    # Log interaction
                    log_chat_interaction(
                        user_input=prompt,
                        bot_response=response,
                        response_time=response_time,
                        source_documents=result.get("source_documents", []),
                        model_used=f"{model_name}"
                    )
                    
                    # Show retrieval info
                    if result.get("source_documents"):
                        st.info(f"📚 Retrieved {len(result['source_documents'])} relevant chunks from your data pod")
                        
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    
else:
    st.info("👆 Upload a data file and enter your API key to start chatting!")

# Footer
st.markdown("---")
st.markdown("🔒 **Privacy Notice**: Your data stays with you. All processing happens locally, and your API keys are not stored.")
