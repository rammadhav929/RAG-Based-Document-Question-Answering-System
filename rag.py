import fitz
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from llm import llm


# ⭐ global files for persistence
INDEX_FILE = "resume.index"
CHUNK_FILE = "chunks.npy"

# ⭐ load embedding model once
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ⭐ global runtime variables
index = None
chunks = None


# ---------- LOAD PDF ----------
def load_pdf(path):
    text = ""
    doc = fitz.open(path)

    # extract text from all pages
    for page in doc:
        text += page.get_text()

    return text


# ---------- SEMANTIC CHUNK ----------
def chunk_text(text, size=180, overlap=40):
    words = text.split()
    chunks = []

    # sliding window chunking
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)

    return chunks


# ---------- BUILD / REBUILD INDEX ----------
def build_index(pdf_path):

    global index, chunks

    # ⭐ extract resume text
    text = load_pdf(pdf_path)

    # ⭐ create chunks
    chunks = chunk_text(text)

    print("TOTAL CHUNKS:", len(chunks))

    # ⭐ generate embeddings
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    # ⭐ build FAISS vector index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # ⭐ save index + chunks (persistence)
    faiss.write_index(index, INDEX_FILE)
    np.save(CHUNK_FILE, chunks)

    print("Index built and saved.")


# ---------- LOAD INDEX (ON SERVER START) ----------
def load_index():

    global index, chunks

    if os.path.exists(INDEX_FILE):

        print("Loading existing FAISS index...")

        index = faiss.read_index(INDEX_FILE)
        chunks = np.load(CHUNK_FILE, allow_pickle=True)

    else:
        print("No index found. Upload resume first.")


# ---------- RAG QUERY ----------
def rag_query(query, k=5):

    global index, chunks

    if index is None:
        return {"answer": "No resume indexed yet.", "sources": []}

    expanded_query = query
    # encode query
    q_emb = embedder.encode([expanded_query], convert_to_numpy=True)

    # safe k value
    k = min(k, len(chunks))

    # search vector DB
    D, I = index.search(q_emb, k)

    retrieved = [chunks[i] for i in I[0]]

    context = "\n\n".join(retrieved)

    # ⭐ grounded prompt
    prompt = f"""
Use ALL context carefully.
Give complete answer.

Context:
{context}

Question: {query}
Answer:
"""

    answer = llm(prompt)
    print(answer)

    return {
        "answer": answer,
        "sources": retrieved
    }


# ⭐ load index automatically when module imported
load_index()