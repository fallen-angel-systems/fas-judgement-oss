"""
Scanner Sub-Package
-------------------
WHY: The scanner handles the core attack execution loop — picking patterns,
     sending them to the target, scoring responses, and streaming results
     back via SSE. Split into service (orchestration), scorer (classification),
     and runner (HTTP execution) for testability.
"""
