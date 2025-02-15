from fastapi.testclient import TestClient

from ai_vs_human_text.api.app import create_app


def test_health() -> None:
    with TestClient(create_app()) as client:
        r = client.get("/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_classify() -> None:
    with TestClient(create_app()) as client:
        r = client.post("/v1/classify", json={"text": "idk lol maybe later"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in ("human", "ai")
    assert "input_char_count" in body


def test_classify_validation_empty() -> None:
    with TestClient(create_app()) as client:
        r = client.post("/v1/classify", json={"text": "   "})
    assert r.status_code == 422
