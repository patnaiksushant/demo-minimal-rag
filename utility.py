from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import Optional

def load_embedder(model="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text): break
        start = end - overlap
    return chunks

def embed_texts(texts, embedder):
    return embedder.encode(texts, normalize_embeddings=True)

def init_chromadb(persist_path: Optional[str] = None):
    # Use PersistentClient for disk persistence, Client for in-memory
    if persist_path:
        client = chromadb.PersistentClient(path=persist_path, settings=Settings(anonymized_telemetry=False))
    else:
        client = chromadb.Client(settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection("docs")
    return client, collection

def query_chromadb(collection, query, embedder, top_k=4):
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    hits = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        hits.append({"document": doc, "metadata": meta, "distance": dist})
    return hits
