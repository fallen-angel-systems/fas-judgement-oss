"""
FAS Judgement OSS — CLI Entry Point
--------------------------------------
WHY: __main__.py makes `python -m fas_judgement` work, and the pyproject.toml
     entry_points make `judgement` the CLI command.

     Commands:
       judgement                → serve (default)
       judgement serve          → start the web server
       judgement activate <KEY> → activate an Elite license
       judgement deactivate     → remove the license
       judgement status         → show license + pattern info
       judgement demo           → run a quick local demo attack

     WHY argparse (not Click): keeps the dependency list minimal for a tool
     that's often installed in constrained environments.

LAYER: CLI — imports everything (config, http, utils).
"""

import argparse
import json
import sys

from .config import BANNER


# === SECTION: UPDATE CHECKER === #

def _parse_version(v: str):
    """Parse version string into tuple of ints for proper comparison."""
    try:
        return tuple(int(x) for x in v.split("."))
    except (ValueError, AttributeError):
        return (0,)

def check_for_updates(current_version: str) -> None:
    """
    Ping PyPI for a newer version. Silent on failure (no network = no problem).
    WHY: Helps users know when to upgrade without requiring a separate tool.
    """
    try:
        import urllib.request
        resp = urllib.request.urlopen(
            "https://pypi.org/pypi/fas-judgement/json", timeout=3
        )
        data = json.loads(resp.read())
        latest = data["info"]["version"]
        if _parse_version(latest) > _parse_version(current_version):
            print(
                f"\n  \033[33m⚡ Update available: {current_version} → {latest}\033[0m"
            )
            print(f"  \033[33m   Run: pip install --upgrade fas-judgement\033[0m\n")
        else:
            print(f"  Version {current_version} (up to date)")
    except Exception:
        print(f"  Version {current_version}")


# === SECTION: DEMO COMMAND === #

def run_demo():
    """
    Run a quick local demo attack against the built-in demo target.

    WHY: Lets new users verify the tool works without needing an external AI
    endpoint. The demo target is a simple mock that responds predictably.
    """
    print(BANNER)
    print("  Running demo attack against local mock target...")
    print("  This tests the scanner against a predictable endpoint.\n")

    try:
        from .modules.ai_security.demo.target import DemoTarget
        from .modules.ai_security.scanner.service import AttackService
        import asyncio

        async def _demo():
            # Start demo target (in-process mock)
            target = DemoTarget()
            await target.start()
            print(f"  Demo target running at: {target.url}")
            print("  Running 5 patterns...\n")

            # Run a quick attack (5 patterns, demo target)
            service = AttackService()
            results = await service.run_attack(
                target_url=target.url,
                method="POST",
                payload_field="message",
                patterns_limit=5,
            )

            for r in results:
                verdict_icon = {"BLOCKED": "✅", "BYPASS": "❌", "PARTIAL": "⚠️", "ERROR": "💥"}.get(r["verdict"], "?")
                print(f"  {verdict_icon} [{r['verdict']}] {r['pattern_text'][:60]}")

            await target.stop()
            print("\n  Demo complete. Run 'judgement' to start the full UI.")

        asyncio.run(_demo())
    except ImportError as e:
        # Demo target may not exist in all builds — graceful fallback
        print(f"  Demo mode requires additional dependencies: {e}")
        print("  Try 'judgement serve' to start the web UI instead.")
    except Exception as e:
        print(f"  Demo error: {e}")
        print("  Try 'judgement serve' to start the web UI instead.")


# === SECTION: SERVE COMMAND === #

def run_server(host: str, port: int) -> None:
    """Start the Judgement web server."""
    import uvicorn
    from .utils.license import startup_check, load_patterns

    print(BANNER)
    from fas_judgement import __version__
    check_for_updates(__version__)

    # License check and pattern sync
    print("  Checking license...")
    license_status = startup_check()
    tier = license_status.get("tier", "free")
    tier_display = tier.replace("_", " ").title()
    print(f"  Tier: {tier_display}")
    print(f"  {license_status.get('message', '')}")
    print()

    # Load patterns (post-license-check so we get the right count)
    patterns = load_patterns()

    print(f"  Starting Judgement on http://localhost:{port}")
    print(f"  Patterns loaded: {len(patterns):,}")
    print()

    from .http.app import app
    uvicorn.run(app, host=host, port=port, log_level="info")


# === SECTION: MAIN === #

def main() -> None:
    """Entry point for the `judgement` CLI command."""
    from .utils import license as license_client

    parser = argparse.ArgumentParser(
        prog="judgement",
        description="FAS Judgement OSS — Prompt Injection Attack Console",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  judgement                       # Start the web UI (default port 8668)
  judgement --port 9000           # Start on a custom port
  judgement activate FAS-XXXX-XXXX-XXXX-XXXX  # Unlock Elite patterns
  judgement deactivate            # Revert to free tier
  judgement status                # Show license & pattern info
  judgement demo                  # Quick local demo
        """,
    )

    # Top-level flags (apply when no subcommand is given → serve mode)
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8668,
        help="Port to bind to (default: 8668)",
    )

    subparsers = parser.add_subparsers(dest="command")

    # serve (explicit)
    serve_parser = subparsers.add_parser("serve", help="Start the web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8668)

    # activate
    activate_parser = subparsers.add_parser("activate", help="Activate a license key")
    activate_parser.add_argument(
        "key", help="License key (FAS-XXXX-XXXX-XXXX-XXXX)"
    )

    # deactivate
    subparsers.add_parser("deactivate", help="Deactivate current license")

    # status
    subparsers.add_parser("status", help="Show license and pattern status")

    # demo
    subparsers.add_parser("demo", help="Run a quick local demo attack")

    args = parser.parse_args()

    # Route to the right handler
    if args.command == "activate":
        ok = license_client.activate(args.key)
        sys.exit(0 if ok else 1)

    elif args.command == "deactivate":
        license_client.deactivate()
        sys.exit(0)

    elif args.command == "status":
        license_client.status()
        sys.exit(0)

    elif args.command == "demo":
        run_demo()
        sys.exit(0)

    elif args.command == "serve":
        run_server(host=args.host, port=args.port)

    else:
        # No subcommand → default serve
        run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
