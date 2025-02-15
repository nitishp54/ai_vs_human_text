from pydantic import BaseModel, Field

from ai_vs_human_text.agent.tool_repair import repair_and_validate


class SampleArgs(BaseModel):
    page: int = Field(..., ge=1)


def test_repair_stringified_int() -> None:
    out = repair_and_validate(SampleArgs, {"page": "3"})
    assert out.page == 3


def test_repair_passes_clean() -> None:
    out = repair_and_validate(SampleArgs, {"page": 2})
    assert out.page == 2
