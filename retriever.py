"""
retriever.py
------------
Loads a saved FAISS index and retrieves the most relevant chunks for a query.

Design decision: top_k = 4. Small enough to keep the prompt focused (and
cheap), large enough to usually cover the answer even if it spans more than
one chunk. This is the kind of parameter worth tuning against a real eval
set rather than leaving at a guessed default in a production system.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ingest import INDEX_DIR, get_embeddings

TOP_K = 4
# Below this similarity score, we treat the result as "not found" rather
# than forcing the LLM to answer from weak/irrelevant context.
SCORE_THRESHOLD = 0.75


def retrieve(doc_id: str, query: str, top_k: int = TOP_K) -> list[tuple[Document, float]]:
    """
    Returns a list of (chunk, relevance_score) tuples, most relevant first.
    relevance_score is 0-1 similarity (higher = more relevant).
    """
    index_path = INDEX_DIR / doc_id
    if not index_path.exists():
        raise FileNotFoundError(f"No index found for doc_id '{doc_id}'. Upload it first.")

    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )

    # similarity_search_with_relevance_scores normalizes distance to a 0-1
    # "higher is better" score, which is easier to threshold than raw L2 distance.
    results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)

    # Filter out weak matches instead of blindly passing everything to the LLM.
    filtered = [(doc, score) for doc, score in results if score >= SCORE_THRESHOLD]
    return filtered if filtered else results[:1]  # keep at least the top hit for context
