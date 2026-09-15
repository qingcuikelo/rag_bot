"""单元测试：意图分类与路由映射。"""

from __future__ import annotations

from xingchi_rag.graph.intent import classify_intent, is_pii_request, route_for_intent


def test_classify_intents() -> None:
    assert classify_intent("XC-L100 库存多少") == "price_stock"
    assert classify_intent("智能门锁保修几年") == "policy_warranty"
    assert classify_intent("门锁连不上WiFi怎么办") == "install_debug"
    assert classify_intent("XC-L100 支持哪些解锁方式") == "product_spec"
    assert classify_intent("我的工单处理到哪了") == "ticket_order"
    assert classify_intent("你好") == "chitchat"
    assert classify_intent("支持以旧换新吗") == "unknown"


def test_pii_request_detected_as_ticket() -> None:
    assert is_pii_request("帮我查一下张伟的联系电话") is True
    assert classify_intent("帮我查一下张伟的联系电话") == "ticket_order"


def test_route_mapping() -> None:
    assert route_for_intent("product_spec") == "knowledge"
    assert route_for_intent("price_stock") == "price_stock"
    assert route_for_intent("ticket_order") == "structured"
    assert route_for_intent("chitchat") == "chitchat"
    assert route_for_intent("unknown") == "unknown"
