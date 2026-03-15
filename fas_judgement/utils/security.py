"""
Security Utilities
------------------
WHY: SSRF protection, input sanitisation, and cURL parsing are pure functions
     with no side effects — ideal for a utils module. Keeping them here means
     both the scanner service and the HTTP route handlers can reuse the same
     validation logic without circular imports.

LAYER: Utils (infrastructure) — imports only stdlib.

SOURCE: Extracted from server.py lines 187–190, 593–718.
"""

import html
import ipaddress
import json
import re
import shlex
import socket
from urllib.parse import urlparse


# === SECTION: INPUT SANITISATION === #

def sanitize_text(text: str) -> str:
    """
    Sanitize user input to prevent stored XSS.

    WHY HTML-escape rather than strip: we want to preserve the content of
    attack payloads for display but neutralise any embedded HTML tags.
    The `quote=True` flag also escapes double-quotes (important for attributes).
    """
    return html.escape(text, quote=True)


# === SECTION: SSRF PROTECTION === #

def is_safe_url(url: str) -> bool:
    """
    Check that a URL does NOT point to private/internal addresses.

    WHY this matters for a security scanner: without SSRF protection,
    an attacker could use our scanner to probe internal services by
    supplying `target_url=http://169.254.169.254/` (AWS metadata) etc.

    Strategy:
      1. Parse the URL and check scheme is http/https
      2. Block known dangerous hostnames by name
      3. Resolve the hostname to IP and reject private/loopback/reserved ranges
      4. If DNS resolution fails, block (unknown = dangerous)
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.scheme not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    # Block known dangerous hostnames by name (DNS rebinding resilience)
    blocked_hosts = {"localhost", "metadata.google.internal"}
    if hostname.lower() in blocked_hosts:
        return False

    # Resolve hostname to IP and check every resolved address
    try:
        infos = socket.getaddrinfo(hostname, None)
        for info in infos:
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
            # Explicitly block 0.0.0.0 (some OS quirks return this)
            if str(ip) == "0.0.0.0":
                return False
    except (socket.gaierror, ValueError):
        # DNS resolution failed → block (principle of least privilege)
        return False

    return True


# === SECTION: CURL PARSER === #

def parse_curl_command(curl_str: str) -> dict:
    """
    Parse a cURL command string and extract URL, method, headers, and body.

    WHY: Users paste cURL commands from browser DevTools as a fast way
    to configure a target. This function translates those into the fields
    our scan form needs (url, method, headers, payload field, template).

    Returns a dict with keys: url, method, headers, field, payload_template.
    Raises ValueError on parse failure.
    """
    curl_str = curl_str.strip()
    # Handle line continuations (\<newline>)
    curl_str = curl_str.replace("\\\n", " ").replace("\\\r\n", " ")

    try:
        tokens = shlex.split(curl_str)
    except ValueError:
        raise ValueError("Could not parse cURL command")

    if not tokens or tokens[0] != "curl":
        raise ValueError("Command must start with 'curl'")

    url = ""
    method = "GET"
    headers = {}
    data = ""
    method_set = False

    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("-X", "--request"):
            i += 1
            if i < len(tokens):
                method = tokens[i].upper()
                method_set = True
        elif tok in ("-H", "--header"):
            i += 1
            if i < len(tokens):
                hval = tokens[i]
                if ":" in hval:
                    key, val = hval.split(":", 1)
                    headers[key.strip()] = val.strip()
        elif tok in ("-d", "--data", "--data-raw", "--data-binary"):
            i += 1
            if i < len(tokens):
                data = tokens[i]
            if not method_set:
                method = "POST"
        elif not tok.startswith("-") and not url:
            url = tok
        i += 1

    if not url:
        raise ValueError("No URL found in cURL command")

    # ---- Payload field detection ---- #
    # We detect which JSON field carries the user message so we can
    # replace it with {{PAYLOAD}} in the template.
    field = ""
    payload_template = ""
    ai_fields = ["messages", "prompt", "text", "message", "input", "query"]

    if data:
        try:
            body = json.loads(data)
            if isinstance(body, dict):
                for f in ai_fields:
                    if f in body:
                        field = f
                        # Replace user content with the {{PAYLOAD}} placeholder
                        if f == "messages" and isinstance(body[f], list):
                            for msg in body[f]:
                                if isinstance(msg, dict) and msg.get("role") == "user":
                                    msg["content"] = "{{PAYLOAD}}"
                                    break
                        else:
                            body[f] = "{{PAYLOAD}}"
                        payload_template = json.dumps(body)
                        break
                if not field:
                    payload_template = data
        except (json.JSONDecodeError, TypeError):
            payload_template = data

    return {
        "url": url,
        "method": method,
        "headers": headers,
        "field": field or "message",
        "payload_template": payload_template,
    }
