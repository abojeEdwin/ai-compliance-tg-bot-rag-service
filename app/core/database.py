import os
import asyncpg
from core.logger import get_logger

logger = get_logger(__name__)

db_pool = None


async def init_db_pool():
    global db_pool
    if not db_pool:
        db_url = os.getenv("DATABASE_URL")
        logger.info("Connecting to PostgreSQL database...")
        db_pool = await asyncpg.create_pool(db_url)
        logger.info("asyncpg connection pool created successfully.")


async def close_db_pool():
    global db_pool
    if db_pool:
        logger.info("Closing asyncpg connection pool...")
        await db_pool.close()
        db_pool = None
        logger.info("asyncpg connection pool closed.")


def get_db():
    if db_pool is None:
        logger.warning("get_db() called but pool is not initialised!")
    return db_pool
