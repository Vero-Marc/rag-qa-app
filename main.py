"""
main.py
-------
FastAPI app wiring together ingestion, retrieval, and generation behind
two endpoints, plus a minimal static chat UI.

Run with:  uvicorn main:app --reload
Then open: http://127.0.0.1:8000
"""

import shutil
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()  # reads OPENAI_API_KEY from a local .env file

from ingest import UPLOAD_DIR, ingest_pdf, list_indexed_docs  # noqa: E402
from retriever import retrieve  # noqa: E402
from chat import answer_question  # noqa: E402

app = FastAPI(title="RAG Document Q&A")

# Permissive CORS for local demo purposes only - tighten before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    doc_id: str
    question: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_created: int


@app.get("/")
def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/docs-list")
def docs_list():
    """Lets the UI populate a dropdown of previously uploaded documents."""
    return {"doc_ids": list_indexed_docs()}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported right now.")

    doc_id = f"{Path(file.filename).stem}-{uuid.uuid4().hex[:8]}"
    dest_path = UPLOAD_DIR / f"{doc_id}.pdf"

    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        num_chunks = ingest_pdf(str(dest_path), doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    return UploadResponse(doc_id=doc_id, filename=file.filename, chunks_created=num_chunks)


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        chunks = retrieve(req.doc_id, req.question)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    result = answer_question(req.question, chunks)
    return result
