import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from core.database import init_db_pool, close_db_pool
from core.logger import get_logger
from api.routes import router as api_router
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG service — initialising database pool...")
    try:
        await init_db_pool()
        logger.info("Database pool ready.")
    except Exception as exc:
        logger.critical("Failed to initialise database pool: %s", exc, exc_info=True)
        raise

    yield

    logger.info("Shutting down — closing database pool...")
    await close_db_pool()
    logger.info("Database pool closed. Goodbye.")


app = FastAPI(lifespan=lifespan)


# ── Global exception handler so every unhandled error is logged ───────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


app.include_router(api_router, prefix="/api")

logger.info("FastAPI application configured — router mounted at /api")