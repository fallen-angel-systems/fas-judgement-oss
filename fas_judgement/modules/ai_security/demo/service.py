"""
Demo Target Lifecycle Service
-------------------------------
WHY: Manages the demo target subprocess (start/stop/status) as a service,
     separate from the HTTP layer. The router calls this service; the service
     knows nothing about FastAPI or HTTP.

     Keeping process management here (not in the router) means:
       - The CLI `judgement demo-start` can reuse the same logic later
       - Tests can call the service directly without HTTP overhead
       - The router stays thin (DDD: routes → service → infrastructure)

LAYER: Module/Service — business logic + process management, no HTTP imports.
"""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from typing import Optional


# === SECTION: DATA TYPES === #

@dataclass
class DemoStatus:
    """
    Status snapshot of the demo target.

    WHY a dataclass instead of a dict: typed fields prevent typos,
    IDE autocompletion works, and the router's serialisation is explicit.
    """
    running: bool
    managed: bool       # True if we spawned it, False if external
    healthy: bool
    port: int
    persona: Optional[str]
    pid: Optional[int]
    url: str

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "managed": self.managed,
            "healthy": self.healthy,
            "port": self.port,
            "persona": self.persona,
            "pid": self.pid,
            "url": self.url,
        }


@dataclass
class StartResult:
    """Result of a start attempt."""
    success: bool
    already_running: bool
    message: str
    url: str = ""
    persona: str = ""
    port: int = 0
    pid: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "success": self.success,
            "already_running": self.already_running,
            "message": self.message,
        }
        if self.success:
            d["url"] = self.url
            d["persona"] = self.persona
            d["port"] = self.port
            if self.pid is not None:
                d["pid"] = self.pid
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class StopResult:
    """Result of a stop attempt."""
    success: bool
    was_running: bool
    message: str
    pid: Optional[int] = None

    def to_dict(self) -> dict:
        d = {
            "success": self.success,
            "was_running": self.was_running,
            "message": self.message,
        }
        if self.pid is not None:
            d["pid"] = self.pid
        return d


# === SECTION: VALID PERSONAS === #

VALID_PERSONAS = ["hardened", "default", "vulnerable"]


# === SECTION: PROCESS STATE === #

# Module-level singleton state for the demo subprocess.
# WHY module-level: only one demo target should run at a time.
# The subprocess lifecycle is tied to the main Judgement process.
_demo_process: Optional[subprocess.Popen] = None
_demo_persona: str = "default"

DEMO_PORT = 8667


# === SECTION: INTERNAL HELPERS === #

def _is_process_alive() -> bool:
    """Check if the managed demo subprocess is still alive."""
    global _demo_process
    if _demo_process is None:
        return False
    if _demo_process.poll() is None:
        return True
    # Process exited — clean up stale reference
    _demo_process = None
    return False


def _is_port_in_use(port: int = DEMO_PORT) -> bool:
    """
    Check if something is already listening on the demo port.

    WHY: The user might have started `judgement demo` in a terminal
    before clicking the UI button. We detect that to avoid spawning
    a second instance and to show accurate status.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        result = s.connect_ex(("127.0.0.1", port))
        s.close()
        return result == 0
    except Exception:
        return False


def _probe_health() -> tuple[bool, Optional[str]]:
    """
    Hit the demo target's health endpoint.
    Returns (healthy: bool, persona: str | None).
    """
    try:
        req = urllib.request.Request(
            f"http://localhost:{DEMO_PORT}/demo/health",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read().decode())
            return (
                data.get("status") == "ok",
                data.get("active_persona"),
            )
    except Exception:
        return False, None


def _kill_process(proc: subprocess.Popen) -> None:
    """
    Kill a subprocess gracefully (SIGTERM), then force (SIGKILL) if needed.

    WHY SIGTERM first: uvicorn handles SIGTERM for clean shutdown,
    closing sockets and flushing logs.
    """
    pid = proc.pid
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        else:
            proc.terminate()

        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            else:
                proc.kill()
            proc.wait(timeout=2)
    except (ProcessLookupError, OSError):
        # Process already exited — that's fine
        pass


# === SECTION: PUBLIC SERVICE API === #

async def start(persona: str = "default") -> StartResult:
    """
    Start the demo target as a subprocess.

    Flow:
      1. Validate persona
      2. Check if already running (managed or external)
      3. Spawn subprocess
      4. Wait for it to become healthy
      5. Return result

    WHY subprocess (not in-process): The demo is a separate FastAPI app
    with its own uvicorn server. Running it in-process would mean mounting
    it on the main app (mixing auth contexts, shared middleware). A subprocess
    keeps them cleanly isolated — separate port, separate process, no auth.
    """
    global _demo_process, _demo_persona

    # Validate persona
    if persona not in VALID_PERSONAS:
        return StartResult(
            success=False,
            already_running=False,
            message="",
            error=f"Unknown persona '{persona}'. Valid: {VALID_PERSONAS}",
        )

    # Already running (managed by us)?
    if _is_process_alive():
        return StartResult(
            success=True,
            already_running=True,
            message="Demo target is already running.",
            url=f"http://localhost:{DEMO_PORT}/demo/chat",
            persona=_demo_persona,
            port=DEMO_PORT,
            pid=_demo_process.pid,
        )

    # Already running (started externally)?
    if _is_port_in_use():
        return StartResult(
            success=True,
            already_running=True,
            message="Demo target is already running (started externally).",
            url=f"http://localhost:{DEMO_PORT}/demo/chat",
            port=DEMO_PORT,
        )

    _demo_persona = persona

    # Spawn the demo subprocess
    # WHY sys.executable: ensures we use the same venv/Python that's
    # running the main app, so fas_judgement is importable.
    try:
        _demo_process = subprocess.Popen(
            [sys.executable, "-m", "fas_judgement", "demo", persona],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
    except Exception as e:
        _demo_process = None
        return StartResult(
            success=False,
            already_running=False,
            message="",
            error=f"Failed to start demo: {e}",
        )

    # Give uvicorn time to bind the port
    await asyncio.sleep(1.5)

    # Verify it didn't crash on startup
    if not _is_process_alive():
        stderr = ""
        try:
            stderr = _demo_process.stderr.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        _demo_process = None
        return StartResult(
            success=False,
            already_running=False,
            message="",
            error=f"Demo process exited immediately. {stderr}".strip(),
        )

    return StartResult(
        success=True,
        already_running=False,
        message=f"Demo target started ({persona} persona).",
        url=f"http://localhost:{DEMO_PORT}/demo/chat",
        persona=persona,
        port=DEMO_PORT,
        pid=_demo_process.pid,
    )


async def stop() -> StopResult:
    """
    Stop the managed demo subprocess.

    Only kills processes we spawned. If the demo was started externally
    (e.g. `judgement demo` in a terminal), we don't touch it.
    """
    global _demo_process

    if not _is_process_alive():
        _demo_process = None
        return StopResult(
            success=True,
            was_running=False,
            message="Demo target was not running.",
        )

    pid = _demo_process.pid
    _kill_process(_demo_process)
    _demo_process = None

    return StopResult(
        success=True,
        was_running=True,
        message="Demo target stopped.",
        pid=pid,
    )


async def status() -> DemoStatus:
    """
    Get the current demo target status.

    Checks both the managed subprocess AND the port, so we correctly
    report demos started from the CLI.
    """
    managed_alive = _is_process_alive()
    port_in_use = _is_port_in_use()
    running = managed_alive or port_in_use

    healthy = False
    persona = _demo_persona if managed_alive else None
    pid = _demo_process.pid if _demo_process and managed_alive else None

    # If something is on the port, probe health for ground truth
    if running:
        healthy, probed_persona = _probe_health()
        if probed_persona:
            persona = probed_persona

    return DemoStatus(
        running=running,
        managed=managed_alive,
        healthy=healthy,
        port=DEMO_PORT,
        persona=persona,
        pid=pid,
        url=f"http://localhost:{DEMO_PORT}/demo/chat",
    )
