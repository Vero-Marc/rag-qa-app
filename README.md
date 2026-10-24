# RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application: upload a PDF, ask
questions about it, and get answers grounded in the document's actual
content — with sources cited — instead of the model guessing from memory.

## Features

- PDF ingestion: load, clean, and chunk documents for retrieval
- Semantic search over document chunks using FAISS
- Grounded answer generation (explicitly instructed not to hallucinate)
- Source citation — every answer shows which page/chunk it came from
- Relevance-score filtering — if nothing relevant is found, the app says so
  instead of forcing a guess
- Simple web UI (upload + chat) served directly from FastAPI

## Tech Stack

- **Python 3.11+ / FastAPI** — backend API
- **LangChain** — orchestration (document loading, splitting, prompt chain)
- **FAISS** — local vector similarity search
- **OpenAI API** (`text-embedding-3-small` + `gpt-4o-mini`) — embeddings and generation
- Vanilla HTML/JS frontend (no framework needed for this scope)

## Architecture

```
PDF Upload
   │
   ▼
PyPDFLoader (load pages)
   │
   ▼
RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
   │
   ▼
OpenAIEmbeddings  ──────────────►  FAISS index (saved to disk per doc_id)
                                         │
User Question                           │
   │                                    │
   ▼                                    │
Embed question  ──────────────────────► similarity_search_with_relevance_scores (top_k=4)
                                         │
                                         ▼
                              Filter by relevance threshold (>= 0.75)
                                         │
                                         ▼
                      Build grounded prompt (context + question)
                                         │
                                         ▼
                           ChatOpenAI (temperature=0)
                                         │
                                         ▼
                        Answer + cited sources returned to UI
```

## How to Run

```bash
# 1. Clone / cd into the project
cd rag-qa-app

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI API key
cp .env.example .env
# then edit .env and paste your key

# 5. Run the app
uvicorn main:app --reload

# 6. Open the UI
# http://127.0.0.1:8000
```

## Design Decisions (and why)

| Decision | Reasoning |
|---|---|
| **FAISS over Pinecone/OpenSearch** | This is a local, single-user demo — no need for a hosted service, multi-tenant filtering, or network latency. FAISS keeps it dependency-free. |
| **Chunk size 500 / overlap 50** | Balances context completeness against retrieval precision. Large enough to keep a sentence's meaning intact, small enough that irrelevant text doesn't dilute a chunk's embedding. |
| **top_k = 4** | Enough to usually cover an answer that spans more than one chunk, without bloating the prompt (cost + latency) or diluting relevance. |
| **Relevance threshold (0.75) with fallback** | Prevents the LLM from being forced to answer from weak/irrelevant context — directly reduces hallucination risk. |
| **temperature = 0** | This is a factual Q&A task, not creative writing — deterministic, repeatable answers are more trustworthy here. |
| **Explicit "don't know" instruction in the system prompt** | The single highest-leverage line for reducing hallucination in a RAG system — tells the model grounding is mandatory, not optional. |
| **Source citation in every response** | Builds user trust and makes the system auditable — you can verify *why* an answer was given, not just trust it blindly. |

## Known Limitations / Next Steps

- Single document per query (no cross-document search yet)
- No conversational memory — each question is independent, no follow-up context
- No authentication — fine for a local demo, not for public deployment
- Embeddings regenerated per document on upload — no incremental re-indexing
  if a document is updated in place

## Possible v2 Features

- Multi-document search with metadata filtering (by doc, date, type)
- Conversational memory for natural follow-up questions
- Swap OpenAI for AWS Bedrock (Nova Pro) to demonstrate provider flexibility
- Hybrid search (keyword + vector) for exact-match queries (IDs, codes, etc.)
- Dockerize for one-command deployment
  
<img width="973" height="751" alt="image" src="https://github.com/user-attachments/assets/83fd65a6-0d7b-484c-977a-158ccf147197" />
