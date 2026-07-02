from pydantic import BaseModel

class IngestRequest(BaseModel):
    parent_text: str
    child_chunks: list[str]
    metadata: dict = {}

class QueryRequest(BaseModel):
    prompt: str
