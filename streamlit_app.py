import streamlit as st
from utility import chunk_text, load_embedder, embed_texts, init_chromadb, query_chromadb
from llm import generate_answer

st.set_page_config(page_title="RAG with ChromaDB + Ollama", page_icon="📚", layout="wide")

if "embedder" not in st.session_state:
    st.session_state.embedder = load_embedder()
if "chroma_client" not in st.session_state or "chroma" not in st.session_state:
    st.session_state.chroma_client, st.session_state.chroma = init_chromadb(persist_path="./chroma_db")

st.title("📚 Minimal RAG — ChromaDB + Ollama")

st.sidebar.header("⚙️ Settings")
chunk_method_label = st.sidebar.selectbox(
    "Chunking method",
    ["Character (sliding window)", "Sentence-aware"],
    index=0
)
chunk_method = "char" if "Character" in chunk_method_label else "sentence"

chunk_size = st.sidebar.slider("Chunk size", 300, 1000, 500)
overlap = st.sidebar.slider("Chunk overlap", 0, 200, 50)
ollama_model = st.sidebar.text_input("Ollama model", value="llama3.1:8b")
top_k = st.sidebar.slider("Top-K retrieval", 1, 10, 4)

with st.spinner("Uploading in progress…"):
    uploaded_files = st.file_uploader("Upload at least 3 .txt files", type=["txt"], accept_multiple_files=True)

if uploaded_files and st.button("📥 Build Index"):
    st.session_state.chroma_client.delete_collection("docs")
    st.session_state.chroma_client, st.session_state.chroma = init_chromadb(persist_path="./chroma_db")
    texts, meta = [], []

    for f in uploaded_files:
        content = f.read().decode("utf-8", errors="ignore")
        chunks = chunk_text(content, chunk_size, overlap, method=chunk_method)
        texts.extend(chunks)
        meta.extend([{"source": f.name}] * len(chunks))

    embeddings = embed_texts(texts, st.session_state.embedder)
    st.session_state.chroma.add(
        ids=[f"doc_{i}" for i in range(len(texts))], 
        embeddings=embeddings.tolist(), 
        metadatas=meta, 
        documents=texts
        )
    st.success(f"Indexed {len(texts)} chunks from {len(uploaded_files)} files ✅")

st.divider()
q = st.text_area("Ask a question:")
if st.button("🤖 Query with RAG") and q:
    q_emb = st.session_state.embedder.encode([q], normalize_embeddings=True).tolist()
    results = st.session_state.chroma.query(query_embeddings=q_emb, n_results=3)

    hits = [
        {"document": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

    answer = generate_answer(q, [
        {"document": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ])
    st.write("### 🧠 Answer")
    st.write(answer)
    best_hit = min(hits, key=lambda h: h["distance"])
    st.write(f"**Sources:** {best_hit['metadata']['source']} (score={best_hit['distance']:.4f})")


