from pydantic import BaseModel, Field
from time import time
from typing import Union, List, Optional, Dict, Any, Literal
from uuid import uuid4

from OAI.types.common import UsageStats, CommonCompletionRequest

JsonType = Union[None, int, str, bool, List[Any], Dict[str, Any]]

class ChatCompletionMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionRespChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: str
    message: ChatCompletionMessage


class ChatCompletionStreamChoice(BaseModel):
    # Index is 0 since we aren't using multiple choices
    index: int = 0
    finish_reason: Optional[str]
    delta: Union[ChatCompletionMessage, dict] = {}

class ChatCompletionFunction(BaseModel):
    name: str
    description: Optional[str]
    parameters: Dict[str, Any]  # TODO: make this more specific


class ChatCompletionRequestFunctionCallOption(BaseModel):
    name: str


ChatCompletionRequestFunctionCall = Union[
    Literal["none", "auto"], ChatCompletionRequestFunctionCallOption
]

ChatCompletionFunctionParameters = Dict[str, JsonType]  # TODO: make this more specific

class ChatCompletionToolFunction(BaseModel):
    name: str
    description: Optional[str]
    parameters: ChatCompletionFunctionParameters


class ChatCompletionTool(BaseModel):
    type: Literal["function"]
    function: ChatCompletionToolFunction


class ChatCompletionNamedToolChoiceFunction(BaseModel):
    name: str


class ChatCompletionNamedToolChoice(BaseModel):
    type: Literal["function"]
    function: ChatCompletionNamedToolChoiceFunction


ChatCompletionToolChoiceOption = Union[
    Literal["none", "auto"], ChatCompletionNamedToolChoice
]


# Inherited from common request
class ChatCompletionRequest(CommonCompletionRequest):
    # Messages
    # Take in a string as well even though it's not part of the OAI spec
    messages: Union[str, List[Dict[str, Any]]]
    functions: Optional[List[ChatCompletionFunction]] = []
    function_call: Optional[ChatCompletionRequestFunctionCall] = None
    tools: Optional[List[ChatCompletionTool]] = []
    tool_choice: Optional[ChatCompletionToolChoiceOption] = None
    prompt_template: Optional[str] = None
    add_generation_prompt: Optional[bool] = True


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex}")
    choices: List[ChatCompletionRespChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "chat.completion"
    usage: Optional[UsageStats] = None


class ChatCompletionStreamChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid4().hex}")
    choices: List[ChatCompletionStreamChoice]
    created: int = Field(default_factory=lambda: int(time()))
    model: str
    object: str = "chat.completion.chunk"
