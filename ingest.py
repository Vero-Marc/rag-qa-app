"""
ingest.py
---------
Handles the "document -> chunks -> embeddings -> vector store" pipeline.

Design decisions (worth mentioning in interviews / README):
- Chunk size 500 / overlap 50: a reasonable starting point that balances
  context completeness against retrieval precision. Tune per document type.
- FAISS chosen over a managed vector DB (Pinecone/OpenSearch) because this
  is a local, single-user demo with no need for a hosted service, filtering
  at scale, or multi-tenant access control.
- Embeddings and the vector index are persisted to disk so the app doesn't
  need to re-embed documents on every restart.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings() -> OllamaEmbeddings:
    """Central place to swap embedding providers later (e.g. Bedrock Titan)."""
    # return OpenAIEmbeddings(model="text-embedding-3-small")
    return OllamaEmbeddings(
    model="nomic-embed-text"
)


def ingest_pdf(file_path: str, doc_id: str) -> int:
    """
    Loads a PDF, splits it into chunks, embeds them, and saves a FAISS index.

    Returns the number of chunks created.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # one LangChain Document per PDF page, with page metadata

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Tag every chunk with the source document id and a chunk index so we can
    # cite sources later ("Answer based on doc_id, page X, chunk Y").
    for i, chunk in enumerate(chunks):
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["chunk_index"] = i

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    index_path = INDEX_DIR / doc_id
    vector_store.save_local(str(index_path))

    return len(chunks)


def list_indexed_docs() -> list[str]:
    """Returns the doc_ids that currently have a saved FAISS index."""
    if not INDEX_DIR.exists():
        return []
    return [p.name for p in INDEX_DIR.iterdir() if p.is_dir()]
