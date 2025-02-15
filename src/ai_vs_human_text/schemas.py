from pydantic import BaseModel, Field, field_validator


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100_000)

    @field_validator("text")
    @classmethod
    def strip_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must contain non-whitespace characters")
        return v


class ClassifyResponse(BaseModel):
    label: str
    confidence: float | None = None
    probabilities: dict[str, float] | None = None
    input_char_count: int


class ErrorResponse(BaseModel):
    error: str
    code: str
    details: dict | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class AgentRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=20_000)


class ModelInfoResponse(BaseModel):
    model_path: str
    metrics: dict
