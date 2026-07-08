import uuid
from core.database import get_db
from core.logger import get_logger
from services.ai_service import get_query_embedding, get_bulk_embeddings, generate_answer

logger = get_logger(__name__)


async def ingest_document_service(payload):
    import json
    parent_id = str(uuid.uuid4())
    db_pool = get_db()

    logger.debug(
        "ingest_document_service — parent_id=%s, raw chunks=%d",
        parent_id,
        len(payload.child_chunks),
    )

    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO parent_documents (id, text, metadata) VALUES ($1::uuid, $2, $3::jsonb)",
                parent_id, payload.parent_text, json.dumps(payload.metadata)
            )
            logger.debug("Inserted parent document — parent_id=%s", parent_id)

            child_records = []
            payload.child_chunks = [c for c in payload.child_chunks if c.strip()]

            if payload.child_chunks:
                logger.debug(
                    "Generating bulk embeddings for %d chunks...", len(payload.child_chunks)
                )
                vectors = get_bulk_embeddings(payload.child_chunks)
                logger.debug("Embeddings received — count=%d", len(vectors))

                for chunk, vector in zip(payload.child_chunks, vectors):
                    child_records.append((str(uuid.uuid4()), parent_id, chunk, str(vector)))

            await conn.executemany(
                "INSERT INTO child_vectors (id, parent_id, text, embedding) "
                "VALUES ($1::uuid, $2::uuid, $3, $4::vector)",
                child_records,
            )
            logger.debug("Inserted %d child vectors — parent_id=%s", len(child_records), parent_id)

    logger.info(
        "Document ingested — parent_id=%s, chunks=%d",
        parent_id,
        len(payload.child_chunks),
    )
    return {
        "status": "success",
        "parent_id": parent_id,
        "chunks_ingested": len(payload.child_chunks),
    }


async def rag_query_service(prompt: str):
    db_pool = get_db()

    logger.debug("Generating query embedding for prompt: %r", prompt[:80])
    query_vector = get_query_embedding(prompt)
    logger.debug("Query embedding generated.")

    async with db_pool.acquire() as conn:
        logger.debug("Running vector similarity search (top 3)...")
        rows = await conn.fetch("""
            SELECT p.text FROM child_vectors c
            JOIN parent_documents p ON c.parent_id = p.id
            ORDER BY c.embedding <=> $1
            LIMIT 3;
        """, str(query_vector))

    if not rows:
        logger.warning("Vector search returned no results — returning fallback response.")
        return {"answer": "I don't have access to any documentation yet to answer that query."}

    logger.debug("Vector search returned %d context block(s).", len(rows))
    context_blocks = [row["text"] for row in rows]

    logger.debug("Calling LLM to generate grounded answer...")
    answer = generate_answer(prompt, context_blocks)
    logger.info("LLM answer generated successfully.")

    return {"answer": answer}
