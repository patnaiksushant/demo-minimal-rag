# 🔍 Minimal RAG Prototype

This is a minimal Retrieval-Augmented Generation (RAG) prototype using
- ChromaDB as the vector database 
- SentenceTransformers for text embeddings 
- Ollama to run open-source LLMs locally (e.g., LLaMA 3)
- Streamlit for a simple interactive UI

## 🚀 Features
- Upload **at least 3 `.txt` files** and build an index.
- Configurable **chunk size** and **overlap** for text splitting.
- Embed text using `sentence-transformers`
- storing and searching embeddings with **ChromaDB** using persistent storage (`./chroma_db`).
- Query your documents via **RAG** (retrieval + LLM answer).
- Query using streamlit interface, get context-aware answers from an open-source LLM with similarity score.


## ⚙️ Configurable Settings (Sidebar)
- Chunk size: default 500 tokens (range 300–1000). 
- Overlap: default 50 tokens (range 0–200). 
- Ollama model: default llama3.1:8b. 
- Top-K retrieval: default 4 (range 1–10).

## 📦 Setup
[Download](https://ollama.com/download) and install Ollama
 (supports macOS, Linux, Windows WSL).
```commandline
ollama --version
ollama pull llama3.1:8b
ollama serve
```
Other available models:
- llama3.1:70b (large, heavy hardware needed)
- mistral:7b (fast and lightweight)
- gemma:2b (very small, efficient)

```bash
git clone https://github.com/patnaiksushant/demo-minimal-rag.git
cd demo-minimal-rag
pip install -r requirements.txt
```

## ▶️ How to Run the RAG Bot
1. Start the Streamlit app:
```bash
streamlit run app.py
```
2. Open the UI in your browser (http://localhost:8501). 
3. Upload at least 3 .txt files. 
4. Click 📥 Build Index to chunk, embed, and store them in ChromaDB. 
5. Ask a question in the text area and press 🤖 Query with RAG. 
6. The app will:
   - Retrieve the top-k most relevant text chunks. 
   - Pass them as context to the Ollama model. 
   - Display the generated answer along with the best-matching source document.
