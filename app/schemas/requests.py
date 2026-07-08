from pydantic import BaseModel

from typing import Optional, List, Dict

class IngestRequest(BaseModel):
    parent_text: str
    child_chunks: list[str]
    metadata: dict = {}

class QueryRequest(BaseModel):
    prompt: str
    history: Optional[List[Dict[str, str]]] = []
