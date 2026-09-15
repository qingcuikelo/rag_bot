"""鉴权依赖（文档 §12.7）。"""

from __future__ import annotations

import jwt
from fastapi import Header, HTTPException, status
from loguru import logger

from xingchi_rag.config import get_settings


def verify_service_key(authorization: str | None = Header(default=None)) -> None:
    """服务级鉴权：``Authorization: Bearer <service_key>``。"""
    settings = get_settings()
    if not settings.auth_enabled:
        return
    if not settings.service_api_key:
        logger.warning("AUTH_ENABLED=true 但未配置 SERVICE_API_KEY，已放行（仅限本地开发）")
        return

    expected = f"Bearer {settings.service_api_key}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="服务级鉴权失败",
            headers={"WWW-Authenticate": "Bearer"},
        )


def decode_customer_token(token: str | None) -> str | None:
    """解码客户 JWT，返回 ``customer_id``；无效则抛 401。"""
    if not token:
        return None
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="客户身份令牌无效",
        ) from exc
    return str(payload.get("sub")) if payload.get("sub") else None


def resolve_customer_id(customer_token: str | None) -> str | None:
    """从请求体令牌解析客户身份（空则未鉴权）。"""
    if not get_settings().auth_enabled:
        return None
    return decode_customer_token(customer_token)
