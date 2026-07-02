import os
import asyncpg

db_pool = None

async def init_db_pool():
    global db_pool
    if not db_pool:
        db_pool = await asyncpg.create_pool(os.getenv("DATABASE_URL"))

async def close_db_pool():
    global db_pool
    if db_pool:
        await db_pool.close()
        db_pool = None

def get_db():
    return db_pool
