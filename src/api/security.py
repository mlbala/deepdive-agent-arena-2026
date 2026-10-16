"""API security: headers/CSP, request IDs, body limits, rate limiting, API-key auth.

All middleware here is pure ASGI so Server-Sent Events stream unbuffered.
"""

from __future__ import annotations

import hmac
import secrets
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request

from src.core.exceptions import AuthenticationRequiredError, RateLimitedError

ASGIApp = Callable[[dict[str, Any], Callable[[], Awaitable[dict[str, Any]]], Callable[[dict[str, Any]], Awaitable[None]]], Awaitable[None]]

BASE_HEADERS: list[tuple[bytes, bytes]] = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=(), payment=(), usb=()"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cross-origin-resource-policy", b"same-origin"),
]


def content_security_policy(nonce: str) -> str:
    """Strict CSP for the dashboard: only our own scripts plus the nonce'd Tailwind config."""
    return "; ".join(
        [
            "default-src 'self'",
            f"script-src 'self' 'nonce-{nonce}' https://cdn.tailwindcss.com",
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data:",
            "connect-src 'self'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ]
    )


def new_nonce() -> str:
    return secrets.token_urlsafe(16)


class SecurityMiddleware:
    """Adds security headers + X-Request-ID, and enforces a request body size limit."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        incoming = headers.get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if 0 < len(incoming) <= 64 and incoming.replace("-", "").isalnum() else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        declared = headers.get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
            await self._reject(send, request_id, 413, "request body too large")
            return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise _BodyTooLarge
            return message

        async def secure_send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                existing = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in BASE_HEADERS if k not in existing]
                message["headers"] = [*message.get("headers", []), *extra, (b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, limited_receive, secure_send)
        except _BodyTooLarge:
            await self._reject(send, request_id, 413, "request body too large")

    @staticmethod
    async def _reject(send: Any, request_id: str, status: int, message: str) -> None:
        body = ('{"error":{"code":"request_rejected","message":"' + message + '","details":{}}}').encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json"), (b"x-request-id", request_id.encode()), *BASE_HEADERS],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _BodyTooLarge(Exception):
    pass


class RateLimiter:
    """Thread-safe token buckets keyed by (bucket, client)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._buckets: dict[tuple[str, str], tuple[float, float]] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def check(self, bucket: str, client: str, per_minute: int) -> None:
        now = self._clock()
        rate = per_minute / 60.0
        with self._lock:
            tokens, last = self._buckets.get((bucket, client), (float(per_minute), now))
            tokens = min(float(per_minute), tokens + (now - last) * rate)
            if tokens < 1.0:
                retry_after = max(1, int((1.0 - tokens) / rate) + 1)
                self._buckets[(bucket, client)] = (tokens, now)
                raise RateLimitedError(
                    f"Rate limit exceeded for '{bucket}' ({per_minute}/min). Retry in {retry_after}s.",
                    retry_after=retry_after,
                )
            self._buckets[(bucket, client)] = (tokens - 1.0, now)


def client_identity(request: Request) -> str:
    key = presented_api_key(request)
    if key:
        return "key:" + hmac.new(b"deepdive-rate-limit", key.encode(), "sha256").hexdigest()[:16]
    return "ip:" + (request.client.host if request.client else "unknown")


def presented_api_key(request: Request) -> str | None:
    header = request.headers.get("x-api-key")
    if header:
        return header.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    # EventSource and plain download links cannot set headers: allow the query parameter for GETs only.
    if request.method == "GET":
        return request.query_params.get("api_key")
    return None


def verify_api_key(presented: str | None, accepted: frozenset[str]) -> None:
    if not accepted:
        return
    if not presented or not any(hmac.compare_digest(presented, key) for key in accepted):
        raise AuthenticationRequiredError("A valid DeepDive API key is required (X-API-Key header).")
