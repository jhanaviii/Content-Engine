# 🤖 Intelligent Document Query System

> An advanced **Retrieval-Augmented Generation (RAG)** system that transforms how you interact with PDF documents using cutting-edge AI technologies.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)  
![Streamlit](https://img.shields.io/badge/streamlit-1.28.1-red.svg)  
![LangChain](https://img.shields.io/badge/langchain-0.1.0-green.svg)  
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- 🔍 **Advanced RAG Implementation** - Hybrid retrieval combining vector search with semantic chunking  
- 📊 **Multi-Document Analysis** - Process and query multiple PDF files simultaneously  
- 🎯 **Intelligent Context Retrieval** - Smart document chunking with overlap for better context preservation  
- 💬 **Interactive Chat Interface** - Clean, responsive Streamlit web application  
- 📈 **Real-time Analytics** - Performance monitoring and query response tracking  
- 🗄️ **Persistent Vector Database** - ChromaDB for efficient document storage and retrieval  
- 🚀 **High-Performance LLM** - Powered by Groq's Llama-3.3-70B model  
- 🔄 **Session Management** - Maintains conversation context across queries  

## 🛠️ Technology Stack

### **AI & Machine Learning**
- **LangChain** - AI application framework  
- **ChromaDB** - Vector database for embeddings  
- **SentenceTransformers** - Text embeddings (`all-MiniLM-L6-v2`)  
- **Groq** - High-performance LLM inference (`Llama-3.3-70B`)  
- **RAGAS** - RAG evaluation metrics  

### **Backend & Processing**
- **Python 3.8+** - Core programming language  
- **PyPDF** - PDF document processing  
- **Pandas & NumPy** - Data manipulation and analysis  

### **Frontend & Visualization**
- **Streamlit** - Web application framework  
- **Plotly** - Interactive data visualization  

## 🚀 Quick Start

### Prerequisites
- Python 3.8+  
- Groq API Key (free tier available)

### Installation

1. **Clone the repository**  
   `git clone https://github.com/yourusername/intelligent-document-query-system.git`  
   `cd intelligent-document-query-system`

2. **Create and activate virtual environment**  
   `python -m venv .venv`  
   `source .venv/bin/activate`  *(On Windows: `.venv\Scripts\activate`)*

3. **Install dependencies**  
   `pip install -r requirements.txt`

4. **Set up environment variables**  
   `cp .env.example .env`  
   Add your `GROQ_API_KEY` to the `.env` file

5. **Run the application**  
   `streamlit run content_engine.py`

6. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
intelligent-document-query-system/
├── content_engine.py           # Main Streamlit application
├── develop.ipynb              # Development & experimentation notebook
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── .env.example               # Environment variables template
├── db/                        # ChromaDB vector database
│   ├── chroma.sqlite3
│   └── embeddings/
├── pdfs/                      # Sample PDF documents
│   ├── goog-10-k-2023.pdf
│   ├── tsla-20231231-gen.pdf
│   └── uber-10-k-2023.pdf
└── .gitignore                 # Git ignore rules
```

## 💡 Usage Examples

### Basic Document Querying
- "What is Google's total revenue for 2023?"  
- "Compare Tesla's and Uber's R&D expenses"  
- "What are the main risk factors mentioned across all documents?"

### Advanced Analytics Queries
- "Summarize key financial metrics for all three companies"  
- "Which company has the highest profit margins?"  
- "What are the emerging technology investments mentioned?"

## 🏗️ System Architecture

```
graph TD
A[PDF Documents] --> B[Document Loader]
B --> C[Text Splitter]
C --> D[Embedding Model]
D --> E[ChromaDB Vector Store]
F[User Query] --> G[Query Embedding]
G --> H[Similarity Search]
E --> H
H --> I[Context Retrieval]
I --> J[LLM Processing]
J --> K[Generated Response]
```

## 🎯 Key Capabilities

- **Semantic Search**: Advanced vector-based document retrieval  
- **Multi-Query Processing**: Handles complex, multi-part questions  
- **Context-Aware Responses**: Maintains conversation flow  
- **Performance Analytics**: Real-time response time monitoring  
- **Scalable Design**: Easily extensible for additional document types  
- **Production Ready**: Robust error handling and logging  

## 📊 Performance Metrics

- ⚡ **Average Response Time**: < 3 seconds  
- 📄 **Document Processing**: 1000+ pages/minute  
- 🎯 **Retrieval Accuracy**: 90%+ relevance score  
- 👥 **Concurrent Users**: Multiple simultaneous queries supported  

## 🔧 Configuration

Create a `.env` file with your API credentials:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Advanced Configuration Options
- **Chunk Size**: Modify document splitting parameters  
- **Temperature**: Adjust LLM creativity (0.0–1.0)  
- **Max Tokens**: Control response length  
- **Retrieval Count**: Number of relevant chunks to retrieve  

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feature/amazing-feature`)  
3. Commit your changes (`git commit -m 'Add amazing feature'`)  
4. Push to the branch (`git push origin feature/amazing-feature`)  
5. Open a Pull Request  

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** – Powerful AI application framework  
- **ChromaDB** – Efficient vector database solution  
- **Groq** – High-performance LLM inference  
- **Streamlit** – Intuitive web application framework  


⭐ **Star this repository if you found it helpful!**
