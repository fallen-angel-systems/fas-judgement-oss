"""
FastAPI Application Factory (OSS Edition)
------------------------------------------
WHY: app.py wires together middleware, routers, static files, and startup hooks.
     Keeping everything in a factory function means tests can create a fresh
     app instance without global side effects.

     OSS simplifications vs Elite:
       ❌ No AuthMiddleware — you're running this locally, you ARE the user
       ❌ No RawBodyMiddleware — no Stripe webhooks
       ❌ No SessionMiddleware — no login sessions
       ❌ No OAuth setup
       ✅ Static files from ui/static/
       ✅ Module registry (multi-turn routes auto-registered via ai_security)
       ✅ DB init on startup

LAYER: HTTP (outermost) — imports everything.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from ..config import DB_PATH


# === SECTION: RATE LIMITER === #

limiter = Limiter(key_func=get_remote_address)


# === SECTION: SECURITY HEADERS MIDDLEWARE === #

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers (CSP, X-Content-Type-Options, etc.) to all responses."""
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "font-src 'self'; "
            "media-src 'self'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


# === SECTION: APP INSTANCE === #

app = FastAPI(
    title="FAS Judgement OSS",
    version="3.0.19",
    docs_url=None,      # No public Swagger UI (security hygiene even for local)
    redoc_url=None,
    openapi_url=None,
)

# CORS - localhost only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8668", "http://127.0.0.1:8668", "http://localhost:8666", "http://127.0.0.1:8666"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Security headers (CSP, X-Frame-Options, etc.)
app.add_middleware(SecurityHeadersMiddleware)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# === SECTION: ROUTER MOUNTING === #

def _include_routers():
    """
    Mount all routers. Called once during module load.

    WHY deferred: avoids circular imports at package load time — routers
    import from core, which imports from config, so we can't safely do
    top-level imports here.

    OSS routers:
      - core (tier info, presets, sessions, settings, parse-curl)
      - scan (attack, scan-target, llm-test, mcp-test)
      - patterns (GET /api/patterns, POST /api/patterns/submit)

    NOT included (Elite only):
      - router_auth.py — OAuth login
      - router_stripe.py — payment handling
      - router_admin.py — admin panel
      - router_keys.py — API key management
      - router_license.py — license key server (OSS is a client, not server)
    """
    from .router_core import router as core_router
    from .router_scan import router as scan_router
    from .router_patterns import router as patterns_router
    from .router_demo import router as demo_router
    from .router_progression import router as progression_router

    app.include_router(core_router)
    app.include_router(scan_router)
    app.include_router(patterns_router)
    app.include_router(demo_router)
    app.include_router(progression_router)

    # Register challenge target routes on main app too (so UI can call them directly)
    # Share the progression service instance so challenge routes can enforce level locks
    # and generate completion tokens that the progression router can validate
    from ..modules.ai_security.demo.challenge_target import build_challenge_routes
    from .router_progression import _service as progression_service
    build_challenge_routes(app, progression_service=progression_service)

    # WHY import the package (not a class): the ai_security __init__.py is
    # responsible for calling registry.register(AiSecurityModule()). Importing
    # the package triggers that side effect. We never reference the name.
    import fas_judgement.modules.ai_security  # noqa: F401 — triggers auto-registration

    # Mount any routers that modules registered themselves with.
    from ..core import registry
    for module_router in registry.get_all_routers():
        app.include_router(module_router)


_include_routers()


# === SECTION: STARTUP === #

@app.on_event("startup")
async def startup():
    """Initialise the database on startup (creates tables if missing)."""
    from ..core.repository import init_db
    await init_db()


# === SECTION: STATIC FILES & FRONTEND === #

_STATIC_DIR = Path(__file__).parent.parent / "ui" / "static"

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the OSS frontend SPA."""
    from fas_judgement import __version__
    html_path = _STATIC_DIR / "index.html"
    if html_path.exists():
        html = html_path.read_text()
        # Inject current version into static HTML
        html = html.replace("v3.0.19", f"v{__version__}")
        return HTMLResponse(html)
    return HTMLResponse(
        "<h1>FAS Judgement OSS</h1><p>Frontend not found. Place index.html in ui/static/</p>"
    )


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import FileResponse
    ico_path = _STATIC_DIR / "favicon.ico"
    if ico_path.exists():
        return FileResponse(str(ico_path))
    return JSONResponse({}, status_code=404)
