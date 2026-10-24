"""
chat.py
-------
Builds the grounded prompt and calls the LLM to generate an answer.

Design decision: the system prompt explicitly instructs the model to only
answer from the provided context and to say when it doesn't know - this is
the single highest-leverage line for reducing hallucination in a RAG app.
"""

from langchain_core.documents import Document
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the
provided context from the user's document.

Rules:
- Answer only using the given context. Do not use outside knowledge.
- If the context does not contain the answer, say: "I couldn't find that in the document."
- Keep answers concise and directly address the question.
- Do not make up page numbers, figures, or facts not present in the context.
"""


def get_llm() -> ChatOllama:
    """Central place to swap LLM providers later (e.g. AWS Bedrock / Nova Pro)."""
    # return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return ChatOllama(
    model="llama3.1:8b",
    temperature=0)


def build_context(chunks: list[tuple[Document, float]]) -> str:
    parts = []
    for i, (doc, score) in enumerate(chunks):
        page = doc.metadata.get("page", "unknown")
        parts.append(f"[Source {i + 1} | page {page} | relevance {score:.2f}]\n{doc.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str, chunks: list[tuple[Document, float]]) -> dict:
    """
    Returns {"answer": str, "sources": [...]}.
    `chunks` comes straight from retriever.retrieve().
    """
    if not chunks:
        return {
            "answer": "I couldn't find that in the document.",
            "sources": [],
        }

    context = build_context(chunks)
    llm = get_llm()

    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    response = llm.invoke(messages)

    sources = [
        {
            "page": doc.metadata.get("page", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index"),
            "relevance_score": round(score, 3),
            "preview": doc.page_content[:160] + ("..." if len(doc.page_content) > 160 else ""),
        }
        for doc, score in chunks
    ]

    return {"answer": response.content, "sources": sources}
