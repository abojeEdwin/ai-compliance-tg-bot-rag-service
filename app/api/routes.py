from fastapi import APIRouter, HTTPException, Request
from schemas.requests import IngestRequest, QueryRequest
from services.rag_service import ingest_document_service, rag_query_service
from core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


def is_junk_telemetry(text: str) -> bool:
    """Detects and blocks client-side browser fingerprinting fragments."""
    junk_keywords = [
        "webgl", "canvas-blocking", "screen_resolution",
        "vendor_renderer", "user_agent_string",
    ]
    lowercase_text = text.lower()
    return any(keyword in lowercase_text for keyword in junk_keywords)


@router.post("/ingest")
async def ingest_document(payload: IngestRequest, request: Request):
    """
    Saves the heavy parent text layout block and bulk inserts
    pre-split child text chunks alongside their computed embeddings.
    """
    logger.info(
        "POST /api/ingest — %d chunks received, metadata=%s",
        len(payload.child_chunks),
        payload.metadata,
    )

    # Filter junk before handing off to the service
    original_count = len(payload.child_chunks)
    payload.child_chunks = [
        chunk for chunk in payload.child_chunks if not is_junk_telemetry(chunk)
    ]
    filtered_count = original_count - len(payload.child_chunks)
    if filtered_count:
        logger.debug("Filtered %d junk-telemetry chunk(s) from ingest payload.", filtered_count)

    try:
        result = await ingest_document_service(payload)
        logger.info(
            "Ingest complete — parent_id=%s, chunks_ingested=%s",
            result.get("parent_id"),
            result.get("chunks_ingested"),
        )
        return result
    except Exception as exc:
        logger.error("Ingest failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/rag/ask")
async def rag_endpoint(payload: QueryRequest, request: Request):
    """
    Called by your Node.js orchestrator. Performs vector search,
    grabs parent blocks, and returns the grounded LLM text engine answer.
    """
    logger.info("POST /api/rag/ask — prompt=%r", payload.prompt[:120])

    try:
        result = await rag_query_service(payload.prompt, payload.history)
        logger.info("RAG query completed successfully.")
        return result
    except Exception as exc:
        logger.error("RAG query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
