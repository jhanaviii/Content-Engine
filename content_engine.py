import streamlit as st
import asyncio
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
import json
import time
import os
from typing import List, Dict, Any
import uuid

# Core imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

import logging
from dotenv import load_dotenv

# Configure environment
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Generic Content Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


class GenericContentEngineApp:
    def __init__(self):
        self.setup_logging()
        self.session_id = str(uuid.uuid4())
        self.document_sources = set()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


@st.cache_resource
def load_generic_system():
    """Load the complete generic system with caching"""
    engine = GenericContentEngineApp()

    # Load documents
    loader = PyPDFDirectoryLoader(path='pdfs/', glob="**/*.pdf")
    pdfs = loader.load()

    # Track document sources
    for doc in pdfs:
        source_file = doc.metadata.get('source', '')
        if source_file:
            filename = os.path.basename(source_file)
            engine.document_sources.add(filename)

    # Advanced chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_documents(pdfs)

    # Create hybrid retriever
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory='./db')
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    # Ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )

    return engine, ensemble_retriever, embeddings


@st.cache_resource
def initialize_generic_llm():
    """Initialize LLM with generic configuration"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if GROQ_API_KEY is None:
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1000
    )


# Initialize system
engine, retriever, embeddings = load_generic_system()
llm = initialize_generic_llm()

# GENERIC prompt template - works with any documents
template = """You are an expert document analyst with access to various PDF documents. Your task is to analyze the provided context and answer questions accurately based on the available information.

Context Information:
{context}

Instructions:
1. Analyze the provided context carefully from the available documents
2. Answer the question with specific details, numbers, and facts when available
3. Cite the source document and page number when referencing specific information
4. If comparing multiple entities/companies/topics, clearly distinguish between them
5. If information is insufficient or missing, clearly state what's not available
6. Provide reasoning and evidence for your conclusions
7. Be objective and factual in your analysis

Question: {question}

Detailed Analysis and Answer:"""

prompt_template = PromptTemplate(template=template, input_variables=['question', 'context'])
chain = prompt_template | llm

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "analytics" not in st.session_state:
    st.session_state.analytics = {
        "queries": 0,
        "avg_response_time": 0,
        "document_sources": {},
        "query_history": []
    }

# Sidebar with advanced features
with st.sidebar:
    st.title("📚 Generic Document Engine")

    # Document sources info
    st.subheader("📄 Loaded Documents")
    if engine.document_sources:
        for source in sorted(engine.document_sources):
            st.write(f"• {source}")
    else:
        st.info("No documents found in pdfs/ directory")

    # Model settings
    st.subheader("Model Configuration")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 2000, 1000, 100)

    # Retrieval settings
    st.subheader("Retrieval Settings")
    num_docs = st.slider("Documents to Retrieve", 3, 15, 10)
    vector_weight = st.slider("Vector Search Weight", 0.0, 1.0, 0.7, 0.1)

    # Analytics
    st.subheader("📊 Session Analytics")
    st.metric("Total Queries", st.session_state.analytics["queries"])
    st.metric("Avg Response Time", f"{st.session_state.analytics['avg_response_time']:.2f}s")

    # Export chat history
    if st.button("📥 Export Chat History"):
        chat_data = {
            "session_id": engine.session_id,
            "timestamp": datetime.now().isoformat(),
            "document_sources": list(engine.document_sources),
            "messages": st.session_state.messages,
            "analytics": st.session_state.analytics
        }
        st.download_button(
            "Download JSON",
            json.dumps(chat_data, indent=2),
            f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json"
        )

# Main interface
st.title("📚 Generic Content Engine")
st.markdown("*Advanced RAG system that works with any PDF documents*")

# Display document info
if engine.document_sources:
    st.info(f"📄 Loaded {len(engine.document_sources)} document(s): {', '.join(sorted(engine.document_sources))}")
else:
    st.warning("⚠️ No documents found. Please add PDF files to the 'pdfs/' directory.")

# Tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "🔍 Document Explorer", "⚙️ System Info"])

with tab1:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message("user"):
            st.write(message['user'])
        with st.chat_message("assistant"):
            st.write(message['bot'])
            if 'metadata' in message:
                with st.expander("📄 Sources & Metadata"):
                    st.json(message['metadata'])

    # Chat input with dynamic placeholder
    placeholder_text = "Ask me anything about the loaded documents..." if engine.document_sources else "Please add PDF files to the pdfs/ directory first"
    query = st.chat_input(placeholder=placeholder_text, disabled=not engine.document_sources)

    if query and engine.document_sources:
        start_time = time.time()

        # Add user message
        st.session_state.messages.append({"user": query, "bot": "", "metadata": {}})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing documents and generating response..."):
                try:
                    # Retrieve documents with metadata
                    docs = retriever.invoke(query)
                    context = "\n".join([d.page_content for d in docs])

                    # Generate response
                    response = chain.invoke({"question": query, "context": context})

                    # Process response
                    answer = response.content.strip()
                    if answer.startswith("Detailed Analysis and Answer:"):
                        answer = answer[29:].strip()

                    # Calculate response time
                    response_time = time.time() - start_time

                    # Create metadata
                    metadata = {
                        "response_time": f"{response_time:.2f}s",
                        "documents_used": len(docs),
                        "sources": [{"page": doc.metadata.get("page", "N/A"),
                                     "source": doc.metadata.get("source", "N/A")} for doc in docs],
                        "timestamp": datetime.now().isoformat(),
                        "model_config": {
                            "temperature": temperature,
                            "max_tokens": max_tokens
                        }
                    }

                    # Display response
                    st.write(answer)

                    # Update session state
                    st.session_state.messages[-1]["bot"] = answer
                    st.session_state.messages[-1]["metadata"] = metadata

                    # Update analytics
                    st.session_state.analytics["queries"] += 1
                    st.session_state.analytics["avg_response_time"] = (
                            (st.session_state.analytics["avg_response_time"] * (
                                        st.session_state.analytics["queries"] - 1) + response_time)
                            / st.session_state.analytics["queries"]
                    )
                    st.session_state.analytics["query_history"].append({
                        "query": query,
                        "response_time": response_time,
                        "timestamp": datetime.now().isoformat()
                    })

                    # Show metadata
                    with st.expander("📄 Sources & Metadata"):
                        st.json(metadata)

                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages[-1]["bot"] = error_msg

with tab2:
    st.subheader("📊 Advanced Analytics")

    if st.session_state.analytics["query_history"]:
        # Response time chart
        df = pd.DataFrame(st.session_state.analytics["query_history"])
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        fig = px.line(df, x='timestamp', y='response_time',
                      title='Response Time Over Time',
                      labels={'response_time': 'Response Time (s)', 'timestamp': 'Time'})
        st.plotly_chart(fig, use_container_width=True)

        # Query statistics
        st.subheader("Query Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", len(df))
        with col2:
            st.metric("Avg Response Time", f"{df['response_time'].mean():.2f}s")
        with col3:
            st.metric("Max Response Time", f"{df['response_time'].max():.2f}s")
    else:
        st.info("No queries yet. Start chatting to see analytics!")

with tab3:
    st.subheader("🔍 Document Explorer")

    # Document search
    search_term = st.text_input("Search in documents:")
    if search_term and engine.document_sources:
        docs = retriever.invoke(search_term)
        st.write(f"Found {len(docs)} relevant documents")

        for i, doc in enumerate(docs):
            with st.expander(f"Document {i + 1} - Page {doc.metadata.get('page', 'N/A')}"):
                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                st.json(doc.metadata)

with tab4:
    st.subheader("⚙️ System Information")

    system_info = {
        "Session ID": engine.session_id,
        "Model": "llama-3.3-70b-versatile",
        "Embedding Model": "all-MiniLM-L6-v2",
        "Retrieval Method": "Hybrid (Vector + BM25)",
        "Vector Database": "ChromaDB",
        "Document Sources": len(engine.document_sources),
        "Current Temperature": temperature,
        "Max Tokens": max_tokens,
        "System Status": "🟢 Online" if engine.document_sources else "🟡 No Documents"
    }

    for key, value in system_info.items():
        st.write(f"**{key}:** {value}")

    # Health check
    if st.button("🔧 Run System Health Check"):
        with st.spinner("Running diagnostics..."):
            try:
                if engine.document_sources:
                    test_query = "What information is available in the documents?"
                    docs = retriever.invoke(test_query)
                    response = chain.invoke({"question": test_query, "context": docs[0].page_content})
                    st.success("✅ All systems operational")
                else:
                    st.warning("⚠️ No documents loaded")
            except Exception as e:
                st.error(f"❌ System error: {e}")

# Footer
st.markdown("---")
st.markdown("*Generic Content Engine v2.0 - Works with any PDF documents*")
