import ollama

def generate_answer(question, retrieved_chunks, model="llama3.1:8b"):
    context = "\n\n".join(
        f"[{i+1}] {h['metadata']['source']}:\n{h['document']}"
        for i, h in enumerate(retrieved_chunks)
    )
    prompt = f"""
You are a helpful assistant. Use ONLY the provided context to answer.
If insufficient, say so.

Context:
{context}

Question:
{question}

Answer:
"""
    resp = ollama.generate(model=model, prompt=prompt)
    return resp.get("response", "").strip()