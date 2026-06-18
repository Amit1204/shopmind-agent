"""
Tool 2: Review Q&A (RAG)
Answers questions about a product using its reviews.
Pipeline: hybrid retrieval (dense + BM25) → cross-encoder reranking → LLM answer.
"""
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi
from services.vector_store import chroma_store
from utils.helpers import confidence_response
from utils.logger import get_logger

logger = get_logger(__name__)

# Prompt template for grounded answer generation
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful product review analyst.
Answer the user's question ONLY using the review excerpts provided below.
If the reviews don't contain enough information, say so clearly.
Always cite which review supports your answer (use review numbers).

Reviews:
{context}"""),
    ("human", "Question: {question}")
])


@tool
def review_qa(product_id: str, question: str) -> dict:
    """
    Answer a specific question about a product using its customer reviews.
    Use this when user asks about product quality, features, complaints, or comparisons.

    Args:
        product_id: The product to query reviews for
        question: Specific question to answer from reviews

    Returns:
        dict with grounded answer, confidence, and supporting review evidence
    """
    logger.info(f"Review QA: product={product_id}, q={question[:60]}")

    # Step 1: Dense retrieval from Chroma
    dense_chunks = chroma_store.query(question, product_id=product_id, top_k=20)

    if not dense_chunks:
        return confidence_response(
            result="No reviews found for this product.",
            confidence=0.0
        )

    # Step 2: BM25 sparse retrieval on same chunks
    texts = [c["text"] for c in dense_chunks]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(question.lower().split())

    # Step 3: Hybrid score — combine dense cosine + BM25 (RRF)
    def rrf(rank, k=60):
        return 1 / (k + rank)

    dense_ranked = sorted(range(len(texts)), key=lambda i: dense_chunks[i]["score"], reverse=True)
    bm25_ranked = sorted(range(len(texts)), key=lambda i: bm25_scores[i], reverse=True)

    hybrid_scores = {}
    for rank, idx in enumerate(dense_ranked):
        hybrid_scores[idx] = hybrid_scores.get(idx, 0) + rrf(rank)
    for rank, idx in enumerate(bm25_ranked):
        hybrid_scores[idx] = hybrid_scores.get(idx, 0) + rrf(rank)

    top_indices = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:5]
    top_chunks = [dense_chunks[i] for i in top_indices]

    # Step 4: Build context for LLM
    context = "\n\n".join([
        f"[Review {i+1}] {chunk['text']}"
        for i, chunk in enumerate(top_chunks)
    ])

    # Step 5: LLM answer (imported here to avoid circular imports)
    from agents.llm_factory import get_llm
    llm = get_llm()
    chain = QA_PROMPT | llm | StrOutputParser()

    try:
        answer = chain.invoke({"context": context, "question": question})
        avg_score = sum(hybrid_scores[i] for i in top_indices) / len(top_indices)
        confidence = min(avg_score * 10, 1.0)  # normalize to [0,1]

        return confidence_response(
            result=answer,
            confidence=confidence,
            source_count=len(top_chunks),
            evidence=[c["metadata"].get("review_id", "") for c in top_chunks]
        )
    except Exception as e:
        logger.error(f"LLM error in review_qa: {e}")
        return confidence_response(result=context[:500], confidence=0.4)
