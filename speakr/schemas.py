"""API schema models."""

from pydantic import BaseModel, Field


class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=300)


class StatusResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str


class TranscriptResponse(BaseModel):
    status: str
    transcript: str = Field(min_length=1, max_length=300)
