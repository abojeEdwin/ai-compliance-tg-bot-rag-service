import os
import cohere
from dotenv import load_dotenv
from core.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

_api_key = os.getenv("COHERE_API_KEY")
if not _api_key:
    logger.warning("COHERE_API_KEY is not set — Cohere calls will fail at runtime.")

co = cohere.ClientV2(api_key=_api_key)
logger.info("Cohere ClientV2 initialised (model: embed-english-v3.0 / command-r7b-12-2024).")


def get_bulk_embeddings(texts: list[str]) -> list[list[float]]:
    """Generates 384-dimension vector float dimensions in bulk arrays using Cohere."""
    if not texts:
        logger.debug("get_bulk_embeddings called with empty list — returning [].")
        return []

    logger.debug("Requesting bulk embeddings for %d text(s)...", len(texts))
    try:
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"],
        )
        result = response.embeddings.float
        logger.debug("Bulk embeddings received — %d vectors returned.", len(result))
        return result
    except Exception as exc:
        logger.error("Cohere bulk embed failed: %s", exc, exc_info=True)
        raise


def get_query_embedding(text: str) -> list[float]:
    """Generates a single search vector optimised for matching user queries."""
    logger.debug("Requesting query embedding for: %r", text[:80])
    try:
        response = co.embed(
            texts=[text],
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"],
        )
        logger.debug("Query embedding received.")
        return response.embeddings.float[0]
    except Exception as exc:
        logger.error("Cohere query embed failed: %s", exc, exc_info=True)
        raise


def generate_answer(question: str, context_chunks: list[str], history: list = None) -> str:
    """Passes context blocks to Command-R for a bounded, grounded answer."""
    if history is None:
        history = []

    logger.debug(
        "Calling Cohere chat — question=%r, context_blocks=%d, history_length=%d",
        question[:80],
        len(context_chunks),
        len(history),
    )
    joined_context = "\n---\n".join(context_chunks)

    prompt = (
        "You are an expert AI compliance automation assistant specializing in identity verification workflows.\n"
        "Your objective is to answer technical and business integration questions using your provided documentation reference data.\n\n"

        "CRITICAL CONSTRUCT FOR DEVICE TELEMETRY / ANTI-FRAUD DATA:\n"
        "The referenced documentation contains deep browser and device telemetry parameters (such as WebGL extensions, Canvas data, and hardware flags).\n"
        "If a user asks about these terms, DO NOT give a general graphics or computer science explanation.\n"
        "Instead, explain contextually that these parameters represent device fingerprinting metadata captured inside anti-fraud 'Signal Management' payloads (such as 'Get Signals using ID') to identify automated bots or high-risk devices during a verification lifecycle.\n\n"

        "STRICT EXECUTION BOUNDARIES:\n"
        "1. Formulate your answers relying exclusively on the VERIFIED CONTEXT provided below. Do not assume or extrapolate beyond this text.\n"
        "2. If the context does not contain enough concrete documentation data to fully answer the query, respond exactly with: "
        "'I am sorry, I do not have that specific detail on hand. Let me loop in a team member to assist you.' and stop generating immediately.\n"
        "3. Never state or imply that you are an official corporate representative of Youverify; you are an automated technical knowledge base index.\n\n"

        f"VERIFIED CONTEXT:\n{joined_context}"
    )

    messages = [{"role": "system", "content": prompt}]
    
    # Map Node.js 'chatbot' role to Cohere's native 'assistant' role
    for msg in history:
        role = msg.get("role", "user")
        if role == "chatbot":
            role = "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
        
    messages.append({"role": "user", "content": question})

    try:
        response = co.chat(
            model="command-r7b-12-2024",
            messages=messages,
            temperature=0.0,
        )
        answer = response.message.content[0].text
        logger.debug("Cohere chat response received (%d chars).", len(answer))
        return answer
    except Exception as exc:
        logger.error("Cohere chat failed: %s", exc, exc_info=True)
        raise