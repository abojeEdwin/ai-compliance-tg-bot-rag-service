from fastapi import APIRouter, HTTPException
from schemas.requests import IngestRequest, QueryRequest
from services.rag_service import ingest_document_service, rag_query_service

router = APIRouter()

def is_junk_telemetry(text: str) -> bool:
    """Detects and blocks client-side browser fingerprinting fragments."""
    junk_keywords = ["webgl", "canvas-blocking", "screen_resolution", "vendor_renderer", "user_agent_string"]
    lowercase_text = text.toLowerCase()
    return any(keyword in lowercase_text for keyword in junk_keywords)

@router.post("/ingest")
async def ingest_document(payload: IngestRequest):
    """
    Saves the heavy parent text layout block and bulk inserts
    pre-split child text chunks alongside their computed embeddings.
    """
    child_records = []
    for chunk in payload.child_chunks:
        if is_junk_telemetry(chunk):
            continue
        
    vector = get_embedding(chunk)
    child_records.append((str(uuid.uuid4()), parent_id, chunk, str(vector)))

    if child_records:
            await conn.executemany(
                "INSERT INTO child_vectors (id, parent_id, text, embedding) VALUES ($1, $2, $3, $4)",
                child_records
            )
    try:
        return await ingest_document_service(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rag/ask")
async def rag_endpoint(payload: QueryRequest):
    """
    Called by your Node.js orchestrator. Performs vector search,
    grabs parent blocks, and returns the grounded LLM text engine answer.
    """
    try:
        return await rag_query_service(payload.prompt)
    except Exception as e:
        # If your execution fails here, the Node orchestrator routes to the Error Node
        # TODO: Add proper error handling and logging
        # TODO: Integrate ERROR Node from Node.js application to send error messages to users
        raise HTTPException(status_code=500, detail=str(e))
