"""Coerce common LLM tool-call mistakes before Pydantic validation (production agent pattern)."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def repair_and_validate(model: type[T], raw: dict[str, Any]) -> T:
    """
    Retry validation after best-effort coercion (e.g. stringified numbers).
    Extend field handlers as new tools are added.
    """
    data = dict(raw)
    try:
        return model.model_validate(data)
    except ValidationError as first:
        data = _apply_coercions(model, data, first)
        try:
            return model.model_validate(data)
        except ValidationError:
            raise first from None


def _apply_coercions(
    model: type[BaseModel], data: dict[str, Any], err: ValidationError
) -> dict[str, Any]:
    out = dict(data)
    fields = model.model_fields
    for e in err.errors():
        loc = e.get("loc")
        if not loc or loc[0] not in fields:
            continue
        key = str(loc[0])
        field_info = fields[key]
        if field_info.annotation is int and key in out:
            out[key] = _coerce_int(out[key])
        elif field_info.annotation is float and key in out:
            out[key] = _coerce_float(out[key])
        elif field_info.annotation is str and key in out:
            out[key] = _coerce_str(out[key])
        elif field_info.annotation is bool and key in out:
            out[key] = _coerce_bool(out[key])
    return out


def _coerce_int(v: Any) -> Any:
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        return int(v.strip())
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _coerce_float(v: Any) -> Any:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return v
    return v


def _coerce_str(v: Any) -> Any:
    if v is None:
        return v
    return str(v)


def _coerce_bool(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    return v
