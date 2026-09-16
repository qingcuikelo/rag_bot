"""集成测试：FastAPI 接口（离线路径）。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from xingchi_rag.graph.build import get_graph

pytestmark = pytest.mark.integration


@pytest.fixture
def client(isolated_storage):
    get_graph.cache_clear()
    from xingchi_rag.api.routes import app

    return TestClient(app)


def test_health(client) -> None:
    response = client.get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["collection_name"].startswith("xingchi_kb_v")


def test_chat_chitchat(client) -> None:
    response = client.post("/v1/chat", json={"message": "你好"})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["answer"]
    assert body["route"] == "chitchat"


def test_feedback_accepted(client) -> None:
    response = client.post("/v1/feedback", json={"session_id": "s1", "rating": 5})
    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_handoff_and_resume(client) -> None:
    session_id = "session-handoff-1"
    first = client.post("/v1/chat", json={"message": "支持以旧换新吗？", "session_id": session_id})
    assert first.status_code == 200
    body = first.json()
    assert body["handoff"] is True
    assert body["refused"] is True

    resumed = client.post(
        "/v1/chat/resume", json={"session_id": session_id, "message": "人工回复：暂不支持"}
    )
    assert resumed.status_code == 200
    assert resumed.json()["answer"] == "人工回复：暂不支持"
