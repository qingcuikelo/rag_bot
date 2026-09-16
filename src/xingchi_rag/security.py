"""安全与合规（文档 §8.6、§12.7）。

- Prompt 注入防护：识别“忽略以上指令”等越权请求；
- 输出 PII 过滤：回答中默认脱敏手机号/邮箱/身份证。
"""

from __future__ import annotations

import re

from xingchi_rag.logging import mask_pii

# 注入/越权请求模式（中英文）
INJECTION_PATTERNS: tuple[str, ...] = (
    r"忽略(以上|之前|上述|前面|所有|先前).{0,8}(指令|提示|规则|要求|设定)",
    r"无视(以上|之前|上述|前面|所有).{0,8}(指令|提示|规则|要求)",
    r"ignore\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
    r"(system|系统)\s*prompt",
    r"(泄露|显示|告诉我).{0,6}(系统)?(提示词|指令|prompt)",
    r"开发者模式",
    r"jailbreak",
    r"(假装|扮演)你是",
    r"现在开始你是",
    r"repeat\s+the\s+(words|text)\s+above",
)

_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)

INJECTION_TEXT = (
    "抱歉，我无法执行该请求。我会始终遵循星驰科技的客服规范，只依据知识库为用户提供"
    "产品与售后相关信息。请问有什么可以帮您？"
)


def detect_prompt_injection(text: str) -> bool:
    """检测 Prompt 注入/越权请求。"""
    if not text:
        return False
    return bool(_INJECTION_RE.search(text))


def sanitize_output(text: str) -> str:
    """输出层 PII 脱敏（手机号/邮箱/身份证）。"""
    return mask_pii(text or "")
