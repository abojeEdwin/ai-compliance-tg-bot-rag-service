from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.database import init_db_pool, close_db_pool
from api.routes import router as api_router
from dotenv import load_dotenv

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db_pool()
    yield
    await close_db_pool()

app = FastAPI(lifespan=lifespan)

app.include_router(api_router, prefix="/api")