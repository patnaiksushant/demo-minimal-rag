from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import Optional
import re

def load_embedder(model="sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model)

def _chunk_char(text, chunk_size=500, overlap=50):
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks

def _chunk_sentence(text, chunk_size=500, overlap=50):
    text = text.strip()
    if not text:
        return []

    # Simple sentence splitter (no external downloads)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    curr = []
    curr_len = 0

    def emit_with_overlap():
        nonlocal curr, curr_len
        joined = " ".join(curr).strip()
        if joined:
            chunks.append(joined)
        if overlap <= 0 or not curr:
            curr, curr_len = [], 0
            return
        # keep as many tail sentences as fit within the overlap budget
        tail = []
        acc = 0
        for s in reversed(curr):
            add = (1 if tail else 0) + len(s)  # +1 for space
            if acc + add <= overlap:
                tail.insert(0, s)
                acc += add
            else:
                break
        curr = tail
        curr_len = len(" ".join(curr)) if curr else 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # If a single sentence is longer than chunk_size, break it by chars
        if len(s) > chunk_size:
            # flush current
            if curr:
                emit_with_overlap()
            # char-chunk the long sentence
            char_chunks = _chunk_char(s, chunk_size=chunk_size, overlap=overlap)
            chunks.extend(char_chunks)
            curr, curr_len = [], 0
            continue

        # Add sentence if it fits, else emit and start new chunk
        prospective = (curr_len + (1 if curr else 0) + len(s))
        if prospective <= chunk_size:
            curr.append(s)
            curr_len = prospective
        else:
            emit_with_overlap()
            curr = [s]
            curr_len = len(s)

    if curr:
        chunks.append(" ".join(curr).strip())

    return chunks

def chunk_text(text, chunk_size=500, overlap=50, method="char"):
    if method == "sentence":
        return _chunk_sentence(text, chunk_size=chunk_size, overlap=overlap)
    return _chunk_char(text, chunk_size=chunk_size, overlap=overlap)

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
