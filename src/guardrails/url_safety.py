"""SSRF-safe URL policy for the web retriever.

A URL may be fetched only if:
* scheme is http/https, no embedded credentials, port is 80/443 (or default);
* the host is not localhost / *.local / *.internal / cloud metadata, nor on the blocklist;
* every address the host resolves to is a public (global) unicast address.

The retriever applies the policy to the initial URL AND to every redirect hop, streams
responses with a hard byte cap, and only parses allow-listed content types.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterable
from typing import Any
from urllib.parse import urlsplit

Resolver = Callable[..., list[tuple[Any, ...]]]

_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home.arpa", ".intranet", ".corp")
_BLOCKED_HOSTS = frozenset({"localhost", "metadata.google.internal", "metadata", "instance-data"})
ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/pdf")


def _ip_is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not (ip.is_multicast or ip.is_reserved or ip.is_loopback or ip.is_link_local)


class UrlSafetyPolicy:
    def __init__(self, blocked_domains: Iterable[str] = (), resolver: Resolver | None = None) -> None:
        self.blocked_domains = frozenset(d.lower().removeprefix("www.") for d in blocked_domains)
        self._resolver = resolver or socket.getaddrinfo

    def is_blocked_domain(self, host: str) -> bool:
        host = host.lower().rstrip(".").removeprefix("www.")
        return any(host == d or host.endswith("." + d) for d in self.blocked_domains)

    def check(self, url: str) -> str | None:
        """Return a human-readable reason if the URL must not be fetched, else None."""
        try:
            parts = urlsplit(url.strip())
        except ValueError:
            return "unparseable URL"
        if parts.scheme not in ("http", "https"):
            return f"scheme '{parts.scheme or 'none'}' not allowed"
        if parts.username or parts.password or "@" in parts.netloc:
            return "credentials embedded in URL"
        try:
            port = parts.port
        except ValueError:
            return "invalid port"
        if port not in (None, 80, 443):
            return f"port {port} not allowed"
        host = (parts.hostname or "").lower().rstrip(".")
        if not host:
            return "missing host"
        if host in _BLOCKED_HOSTS or host.endswith(_BLOCKED_HOST_SUFFIXES):
            return f"internal hostname '{host}'"
        if self.is_blocked_domain(host):
            return f"domain '{host}' is on the blocklist"

        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            literal = None
        if literal is not None:
            return None if _ip_is_public(literal) else f"non-public IP address {literal}"

        try:
            infos = self._resolver(host, port or (443 if parts.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
        except (OSError, UnicodeError) as exc:
            return f"DNS resolution failed ({type(exc).__name__})"
        addresses = {info[4][0] for info in infos}
        if not addresses:
            return "host resolved to no addresses"
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address.split("%", 1)[0])
            except ValueError:
                return f"unparseable resolved address {address}"
            if not _ip_is_public(ip):
                return f"host resolves to non-public address {ip}"
        return None
