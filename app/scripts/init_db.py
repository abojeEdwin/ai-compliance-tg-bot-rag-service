import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()

async def init_db():
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS parent_documents (
                id UUID PRIMARY KEY,
                text TEXT NOT NULL,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await conn.execute("""
            DROP TABLE IF EXISTS child_vectors;
            CREATE TABLE child_vectors (
                id UUID PRIMARY KEY,
                parent_id UUID REFERENCES parent_documents(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                embedding vector, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        print("Database schema initialized successfully with pgvector.")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(init_db())