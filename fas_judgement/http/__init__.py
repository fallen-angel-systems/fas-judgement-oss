"""
HTTP Layer
----------
WHY: The HTTP layer is the outermost ring in the DDD onion. It:
     1. Translates HTTP requests into domain calls
     2. Translates domain responses (or errors) into HTTP responses
     3. Handles auth middleware, rate limiting, and CORS

     HTTP layer knows about domain models, but domain models know NOTHING
     about FastAPI, requests, or responses.

LAYER: HTTP / Presentation.
"""
