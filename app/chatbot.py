import logging
import os

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
TOP_K = int(os.getenv("TOP_K", "3"))

PROMPT_TEMPLATE = """
You are an expert Northwestern men's soccer coaching assistant with deep knowledge of:
- Modern attacking and defensive tactics (pressing, build-up, set pieces)
- Player development and match preparation
- Match analysis using Wyscout-style metrics (xG, PPDA, progressive passes, duels, etc.)
- Northwestern's Big Ten context and 2024 season history

Coaching principles:
- Ground every recommendation in the provided context
- Be specific, actionable, and evidence-based
- Reference stats or matches by name when they support the point

Context:
{context}
---
Coach's question:
{question}
---
Provide a clear, structured response:
1. Quick read of the situation
2. Specific recommendations with rationale
3. How to implement in training or the next match
4. Risks or trade-offs to watch for
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("coacheye.chatbot")


def _init_db() -> Chroma:
    embedding_function = OpenAIEmbeddings()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


def _search_db(db: Chroma, query: str):
    results = db.similarity_search_with_relevance_scores(query, k=TOP_K)
    return [
        (doc.page_content, doc.metadata.get("source"), score)
        for doc, score in results
        if score >= RELEVANCE_THRESHOLD
    ]


def create_chatbot():
    """Build a RAG chatbot bound to the persisted Chroma store."""
    db = _init_db()
    model = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def chatbot(inputs):
        query_text = inputs.get("question", "").strip()
        if not query_text:
            return {"answer": "Please ask a question.", "sources": []}

        try:
            results = _search_db(db, query_text)
        except Exception:
            logger.exception("Vector search failed")
            results = []

        logger.info("Retrieved %d relevant chunks", len(results))

        if results:
            context = "\n\n---\n\n".join(chunk for chunk, _, _ in results)
            message = model.invoke(prompt.format(context=context, question=query_text))
            sources = sorted({src for _, src, _ in results if src})
            return {"answer": message.content, "sources": sources}

        fallback = (
            "You are a friendly Northwestern men's soccer coaching assistant. "
            "No indexed documents matched the question below — answer conversationally "
            "and, if useful, ask a clarifying follow-up.\n\n"
            f"Question: {query_text}"
        )
        message = model.invoke(fallback)
        return {"answer": message.content, "sources": []}

    return chatbot
