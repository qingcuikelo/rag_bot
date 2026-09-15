"""日志与 PII 脱敏（loguru）。

对应文档 §2.3.3 / §10.3：日志脱敏后再记录，禁止落原文明文。
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

from loguru import logger

# ---------- PII 脱敏规则 ----------
# 中国大陆手机号（11 位）
_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
# 已脱敏手机号（如 138****6621）
_MASKED_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d)(\*{2,4})(\d{4})(?!\d)")
# 邮箱
_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")
# 身份证（18 位）
_ID_CARD_RE = re.compile(r"(?<!\d)(\d{6})(\d{8})(\d{3}[\dXx])(?!\d)")


def mask_pii(text: str) -> str:
    """对文本中的手机号、邮箱、身份证做脱敏。

    手机号保留前 3 位与后 4 位；邮箱保留首字符与域名；身份证仅保留前 6 位。
    """
    if not text:
        return text

    def _mask_phone(m: re.Match[str]) -> str:
        s = m.group(1)
        return f"{s[:3]}****{s[-4:]}"

    def _mask_masked_phone(m: re.Match[str]) -> str:
        return f"{m.group(1)}****{m.group(3)}"

    def _mask_email(m: re.Match[str]) -> str:
        local, domain = m.group(1), m.group(2)
        head = local[0] if local else "*"
        return f"{head}***@{domain}"

    def _mask_id(m: re.Match[str]) -> str:
        return f"{m.group(1)}********{m.group(3)[-1:]}"

    text = _MASKED_PHONE_RE.sub(_mask_masked_phone, text)
    text = _PHONE_RE.sub(_mask_phone, text)
    text = _EMAIL_RE.sub(_mask_email, text)
    text = _ID_CARD_RE.sub(_mask_id, text)
    return text


def _patcher(record: Any) -> None:
    """loguru patcher：对消息与 extra 做 PII 脱敏。"""
    record["message"] = mask_pii(record["message"])
    extra = record.get("extra")
    if extra:
        for key, value in list(extra.items()):
            if isinstance(value, str):
                extra[key] = mask_pii(value)


class InterceptHandler(logging.Handler):
    """将标准库 logging 转发到 loguru（统一脱敏与格式）。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str = "INFO") -> None:
    """初始化全局日志：控制台输出 + PII 脱敏 + 标准库转发。"""
    logger.remove()
    logger.configure(patcher=_patcher)
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "langchain", "httpx"):
        std_logger = logging.getLogger(name)
        std_logger.handlers = [InterceptHandler()]
        std_logger.propagate = False
