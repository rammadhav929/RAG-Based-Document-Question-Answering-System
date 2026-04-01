# pip install "fastapi[standard]" python-multipart

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from rag import rag_query, build_index, load_index

#fastapi dev api.py to start the project
app = FastAPI()


# ⭐ allow HTML frontend to call API (very important)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for demo
    allow_methods=["*"],
    allow_headers=["*"],
)


# ⭐ load FAISS index when server starts
@app.on_event("startup")
def startup_event():
    load_index()


# ---------- REQUEST MODEL ----------
class Question(BaseModel):
    question: str


# ---------- ASK ENDPOINT ----------
@app.post("/ask")
async def ask_question(q: Question):

    result = rag_query(q.question)

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


# ---------- UPLOAD ENDPOINT ----------
MAX_SIZE = 1024 * 1024 * 1024   # 1GB


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # ⭐ validate extension
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    # ⭐ validate content type (extra safety)
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Invalid file type")

    temp_path = "uploaded_resume.pdf"
    size = 0

    # ⭐ streaming save (memory safe)
    with open(temp_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunk
            if not chunk:
                break

            size += len(chunk)

            if size > MAX_SIZE:
                buffer.close()
                os.remove(temp_path)
                raise HTTPException(400, "File too large (max 1GB)")

            buffer.write(chunk)

    # ⭐ rebuild vector index after upload
    build_index(temp_path)

    return {"message": "PDF uploaded and indexed successfully"}