from pydantic import BaseModel, Field, field_validator


class ClassifyToolArgs(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Raw text to classify as human-written or AI-generated.",
    )

    @field_validator("text", mode="before")
    @classmethod
    def coerce_text(cls, v: object) -> str:
        if v is None:
            raise ValueError("text is required")
        return str(v).strip()


class ModelInfoToolArgs(BaseModel):
    """No fields; LLMs sometimes send empty object or nulls."""

    model_config = {"extra": "ignore"}
