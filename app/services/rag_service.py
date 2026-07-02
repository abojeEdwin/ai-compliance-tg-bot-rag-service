import uuid
from core.database import get_db
from services.ai_service import get_query_embedding, get_bulk_embeddings, generate_answer

async def ingest_document_service(payload):
    import json
    parent_id = str(uuid.uuid4())
    db_pool = get_db()
    
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "INSERT INTO parent_documents (id, text, metadata) VALUES ($1::uuid, $2, $3::jsonb)",
                parent_id, payload.parent_text, json.dumps(payload.metadata)
            )
            
            child_records = []
            payload.child_chunks = [c for c in payload.child_chunks if c.strip()]
            if payload.child_chunks:
                vectors = get_bulk_embeddings(payload.child_chunks)
                for chunk, vector in zip(payload.child_chunks, vectors):
                    child_records.append((str(uuid.uuid4()), parent_id, chunk, str(vector)))
            
            await conn.executemany(
                "INSERT INTO child_vectors (id, parent_id, text, embedding) VALUES ($1::uuid, $2::uuid, $3, $4::vector)",
                child_records
            )
            
    return {"status": "success", "parent_id": parent_id, "chunks_ingested": len(payload.child_chunks)}

async def rag_query_service(prompt: str):
    db_pool = get_db()
    
    # 1. Embed incoming Telegram prompt
    query_vector = get_query_embedding(prompt)
    
    # 2. Query nearest child vectors using cosine distance (<=>) but select parent texts
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT p.text FROM child_vectors c
            JOIN parent_documents p ON c.parent_id = p.id
            ORDER BY c.embedding <=> $1
            LIMIT 3;
        """, str(query_vector))
        
    if not rows:
        return {"answer": "I don't have access to any documentation yet to answer that query."}
        
    context_blocks = [row['text'] for row in rows]
    
    # 3. Generate answer through Cohere
    answer = generate_answer(prompt, context_blocks)
    return {"answer": answer}
