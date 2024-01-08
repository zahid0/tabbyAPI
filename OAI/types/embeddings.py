from typing import List, Union

from pydantic import BaseModel


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    encoding_format: str


class Embedding(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[Embedding]
    model: str
