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
