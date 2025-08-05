# main.py
from fastapi import FastAPI, Request, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv

from app.modules.document_processor import extract_text
from app.modules.text_chunker import chunk_text
from app.modules.embedder import get_embeddings_in_parallel, get_embedding
from app.modules.vector_search import build_and_save_faiss_index, load_faiss_index, faiss_semantic_search
from app.modules.llm import build_prompt, generate_answer

import json
import tempfile

load_dotenv()
API_KEY = os.getenv("API_KEY", "secret")  # fallback for local

app = FastAPI()

# ==== Models ====

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# ==== Core RAG logic ====

def ingest_and_build_index(document_url: str):
    print("📄 Extracting text from document...")
    text = extract_text(document_url)

    print("✂️ Chunking...")
    chunks = chunk_text(text)

    print("💡 Generating embeddings...")
    embeddings = get_embeddings_in_parallel(chunks)

    print("📦 Building FAISS index...")
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = os.path.join(tmpdir, "index.faiss")
        texts_path = os.path.join(tmpdir, "index_texts.json")
        build_and_save_faiss_index(embeddings, chunks, index_path, texts_path)
        index, texts = load_faiss_index(index_path, texts_path)
    
    return index, texts

def query_rag(question: str, index, texts) -> str:
    query_embedding = get_embedding(question)
    top_indices, _ = faiss_semantic_search(index, query_embedding, top_k=3)
    top_chunks = list({texts[i] for i in top_indices})  # deduplicate

    prompt = build_prompt(question, top_chunks)
    answer = generate_answer(prompt)

    return answer

# ==== Endpoint ====

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx(request: Request, data: QueryRequest, authorization: Optional[str] = Header(None)):
    # Auth check
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        # Step 1: Ingest document and build index
        index, texts = ingest_and_build_index(data.documents)

        # Step 2: Answer questions
        answers = []
        for question in data.questions:
            print(f"❓ Processing: {question}")
            answer = query_rag(question, index, texts)
            answers.append(answer)
        
        return QueryResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))



