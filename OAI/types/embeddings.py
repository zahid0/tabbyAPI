from typing import List, Optional, Union

from pydantic import BaseModel

from OAI.types.common import UsageStats

class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = "mymodel"
    encoding_format: Optional[str] = "float"


class Embedding(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Embedding]
    model: str
    usage: Optional[UsageStats] = None
