"""意图分类与路由映射（文档 §6.1）。

规则优先（快、可测）；未命中时返回 ``unknown``，由 LLM 兜底（见节点实现）。
"""

from __future__ import annotations

# 意图 → 路由
INTENT_TO_ROUTE: dict[str, str] = {
    "product_spec": "knowledge",
    "price_stock": "price_stock",
    "policy_warranty": "knowledge",
    "ticket_order": "structured",
    "install_debug": "knowledge",
    "faq_general": "knowledge",
    "chitchat": "chitchat",
    "unknown": "unknown",
}

# PII / 工单类：需要鉴权，归入结构化 SQL 路径
PII_KEYWORDS = ["电话", "手机号", "手机", "姓名", "住在", "地址", "客户信息", "个人信息"]
TICKET_KEYWORDS = ["工单", "订单", "处理到哪", "处理进度", "进度", "tk2026"]
PRICE_KEYWORDS = [
    "多少钱",
    "价格",
    "售价",
    "零售价",
    "出厂价",
    "毛利率",
    "便宜",
    "贵",
    "库存",
    "有货",
    "现货",
    "缺货",
    "补货",
    "发货",
    "在库",
]
POLICY_KEYWORDS = [
    "保修",
    "质保",
    "退换",
    "退货",
    "换新",
    "7天",
    "七天",
    "15天",
    "政策",
    "首响",
    "响应",
    "免责",
    "三包",
]
INSTALL_KEYWORDS = [
    "安装",
    "配网",
    "连接",
    "wifi",
    "wi-fi",
    "指纹失败",
    "识别失败",
    "故障",
    "报修",
    "重置",
    "电池",
    "充电",
    "掉线",
    "连不上",
    "打不开",
    "频段",
    "5ghz",
    "2.4ghz",
]
SPEC_KEYWORDS = [
    "参数",
    "规格",
    "解锁",
    "材质",
    "支持哪些",
    "型号",
    "区别",
    "对比",
    "区别",
    "类别",
    "尺寸",
    "门厚",
    "容量",
    "识别速度",
    "续航",
    "防水",
    "芯片",
]
CHITCHAT_KEYWORDS = ["你好", "您好", "谢谢", "在吗", "hi", "hello", "hey", "哈哈"]

_INTENT_ORDER = [
    ("ticket_order", TICKET_KEYWORDS),
    ("price_stock", PRICE_KEYWORDS),
    ("policy_warranty", POLICY_KEYWORDS),
    ("install_debug", INSTALL_KEYWORDS),
    ("product_spec", SPEC_KEYWORDS),
]


def is_pii_request(question: str) -> bool:
    """是否请求个人信息（需鉴权）。"""
    lowered = question.lower()
    return any(keyword in lowered for keyword in PII_KEYWORDS)


def classify_intent(question: str) -> str:
    """规则意图分类。"""
    if not question:
        return "unknown"
    lowered = question.lower()

    if is_pii_request(question):
        return "ticket_order"

    # “以旧换新”是超纲问法，不应命中退换货政策关键词“换新”
    scan_text = lowered.replace("以旧换新", "")

    for intent, keywords in _INTENT_ORDER:
        if any(keyword in scan_text for keyword in keywords):
            return intent

    if any(keyword in scan_text for keyword in CHITCHAT_KEYWORDS):
        return "chitchat"

    return "unknown"


def route_for_intent(intent: str) -> str:
    """意图 → 路由（knowledge / structured / price_stock / chitchat / unknown）。"""
    return INTENT_TO_ROUTE.get(intent, "unknown")
