"""
FAS Judgement OSS — Local License Client
------------------------------------------
WHY: The OSS app is a license CLIENT, not server. This module handles:
  - Storing the license key in ~/.fas-judgement/config.json
  - Validating the key against the FAS license server
  - Syncing pattern libraries (Elite gets 1000+, free gets 100)
  - CLI commands: activate, deactivate, status

Compare to Elite: the Elite build has utils/license.py that also manages
API keys (jdg_ prefix Bearer tokens). OSS has no API keys — no login,
no multi-user, so those helpers are dropped.

LAYER: Utils (infrastructure) — imports httpx, stdlib, config.
       No imports from HTTP or module layers to avoid circular deps.
"""

import hashlib
import json
import os
import platform
import uuid
from datetime import datetime
from pathlib import Path

import httpx

from ..config import SUBMIT_API_URL


# === SECTION: PATHS & CONSTANTS === #

# WHY ~/.fas-judgement (with dash): avoids collision with the old ~/.judgement
# directory used by the legacy monolith. Users who upgrade keep old cached data
# in the old location; OSS DDD reads from the new location.
CONFIG_DIR = Path.home() / ".fas-judgement"
CONFIG_FILE = CONFIG_DIR / "config.json"
CACHED_PATTERNS_FILE = CONFIG_DIR / "patterns_cache.json"

# Bundled patterns installed with the package (the free 100)
BUNDLED_PATTERNS = (
    Path(__file__).parent.parent
    / "modules/ai_security/patterns/data/single-shot/patterns.json"
)

FREE_PATTERN_COUNT = 100

FAS_API_URL = os.environ.get("FAS_API_URL", SUBMIT_API_URL)


# === SECTION: MACHINE IDENTITY === #

def get_machine_id() -> str:
    """
    Generate a stable machine identifier for seat tracking.

    WHY hash rather than raw MAC: avoids storing PII; the SHA-256 is
    reproducible on the same machine but not reversible to the MAC address.
    """
    parts = [
        platform.node(),
        platform.machine(),
        platform.system(),
        str(uuid.getnode()),  # MAC address as integer
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# === SECTION: CONFIG PERSISTENCE === #

def load_config() -> dict:
    """Load saved config (license key, tier, features) from disk."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(config: dict) -> None:
    """Persist config to ~/.fas-judgement/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def clear_config() -> None:
    """Remove saved config and cached patterns (used on deactivation)."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
    if CACHED_PATTERNS_FILE.exists():
        CACHED_PATTERNS_FILE.unlink()


def get_license_key() -> str | None:
    """
    Get the license key.
    Priority: FAS_LICENSE_KEY env var → saved config file.
    WHY env var first: lets Docker/CI users inject keys without touching config.
    """
    key = os.environ.get("FAS_LICENSE_KEY", "").strip()
    if key:
        return key.upper()
    config = load_config()
    return config.get("license_key")


# === SECTION: LICENSE VALIDATION === #

def validate_license(key: str = None) -> dict:
    """
    Validate a license key against the FAS API.

    Returns a dict like:
      {"valid": True, "tier": "elite_home", "features": {...}}
      {"valid": False, "error": "License invalid", "tier": "free"}

    Side effects:
      - On success: saves key + tier to config (avoids re-validation next startup)
      - On 403: clears saved config (revoked = back to free)
    """
    key = key or get_license_key()
    if not key:
        return {"valid": False, "error": "No license key configured", "tier": "free"}

    machine_id = get_machine_id()

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.post(
                f"{FAS_API_URL}/api/license/validate",
                json={"key": key, "machine_id": machine_id},
            )

            if resp.status_code == 200:
                data = resp.json()
                # Cache the result so startup is faster next time
                config = load_config()
                config["license_key"] = key
                config["tier"] = data.get("tier", "free")
                config["features"] = data.get("features", {})
                config["last_validated"] = datetime.utcnow().isoformat()
                config["expires_at"] = data.get("expires_at")
                save_config(config)
                return data
            elif resp.status_code == 403:
                data = resp.json()
                clear_config()  # Revoked — clear everything
                return {"valid": False, "error": data.get("error", "License invalid"), "tier": "free"}
            else:
                return {"valid": False, "error": f"API error ({resp.status_code})", "tier": "free"}

    except httpx.ConnectError:
        return {"valid": False, "error": "Cannot reach FAS license server", "tier": "free"}
    except httpx.TimeoutException:
        return {"valid": False, "error": "License server timeout", "tier": "free"}
    except Exception as e:
        return {"valid": False, "error": str(e), "tier": "free"}


# === SECTION: PATTERN SYNCING === #

def sync_patterns(key: str = None) -> dict:
    """
    Download the full pattern library from the FAS API and cache locally.

    WHY cache: pattern files can be large (1000+ entries). Caching them
    locally means the app starts instantly on subsequent runs — no network
    dependency unless the license needs re-validation.

    Returns {"success": True, "count": N} or {"success": False, "error": "..."}.
    """
    key = key or get_license_key()
    if not key:
        return {"success": False, "error": "No license key"}

    try:
        with httpx.Client(timeout=60) as client:
            resp = client.get(
                f"{FAS_API_URL}/api/license/patterns",
                headers={"X-License-Key": key},
            )

            if resp.status_code == 200:
                data = resp.json()
                patterns = data.get("patterns", [])

                CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                cache_data = {
                    "patterns": patterns,
                    "synced_at": datetime.utcnow().isoformat(),
                    "version": data.get("version", "unknown"),
                    "count": len(patterns),
                }

                # Multi-turn patterns (Elite only)
                mt = data.get("multi_turn_patterns")
                if mt:
                    cache_data["multi_turn_patterns"] = mt
                    cache_data["multi_turn_count"] = len(mt)

                CACHED_PATTERNS_FILE.write_text(json.dumps(cache_data))

                return {
                    "success": True,
                    "count": len(patterns),
                    "multi_turn_count": len(mt) if mt else 0,
                    "version": data.get("version"),
                }
            elif resp.status_code == 403:
                return {"success": False, "error": "License invalid or expired"}
            else:
                return {"success": False, "error": f"API error ({resp.status_code})"}

    except httpx.ConnectError:
        return {"success": False, "error": "Cannot reach FAS pattern server"}
    except httpx.TimeoutException:
        return {"success": False, "error": "Pattern download timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


# === SECTION: PATTERN LOADING === #

def load_patterns() -> list:
    """
    Load patterns based on license status.

    Priority:
      1. Cached Elite patterns (if valid Elite license + cached file exist)
      2. Bundled free patterns (first 100, always available)

    WHY cap free at 100: the remaining patterns are the value prop for Elite.
    The OSS patterns.json contains all 100 free patterns — no truncation needed
    at the file level, but we apply the cap here defensively.
    """
    config = load_config()
    tier = config.get("tier", "free")

    # Elite tier: use cached patterns downloaded during activation
    if tier in ("elite_home", "elite_business") and CACHED_PATTERNS_FILE.exists():
        try:
            cache = json.loads(CACHED_PATTERNS_FILE.read_text())
            patterns = cache.get("patterns", [])
            if patterns:
                return patterns
        except (json.JSONDecodeError, OSError):
            pass

    # Free tier: use bundled patterns (capped at FREE_PATTERN_COUNT)
    if BUNDLED_PATTERNS.exists():
        try:
            all_patterns = json.loads(BUNDLED_PATTERNS.read_text())
            return all_patterns[:FREE_PATTERN_COUNT]
        except (json.JSONDecodeError, OSError):
            pass

    return []


def load_multi_turn_patterns() -> list:
    """
    Load multi-turn attack patterns from cache (Elite only).
    Free tier gets an empty list — multi-turn requires Elite.
    """
    config = load_config()
    tier = config.get("tier", "free")

    if tier not in ("elite_home", "elite_business"):
        return []

    if CACHED_PATTERNS_FILE.exists():
        try:
            cache = json.loads(CACHED_PATTERNS_FILE.read_text())
            return cache.get("multi_turn_patterns", [])
        except (json.JSONDecodeError, OSError):
            pass

    return []


def get_tier() -> str:
    """Get the current license tier from saved config."""
    return load_config().get("tier", "free")


def get_features() -> dict:
    """Get current feature flags from saved config."""
    return load_config().get("features", {
        "patterns": True,
        "multi_turn": False,
        "teams": False,
        "campaigns": False,
        "reports": False,
        "custom_patterns": False,
        "max_users": 1,
    })


def is_feature_enabled(feature: str) -> bool:
    """Check if a specific feature is enabled for the current tier."""
    return get_features().get(feature, False)


# === SECTION: STARTUP CHECK === #

def startup_check() -> dict:
    """
    Run on app startup: validate license and sync patterns if needed.

    WHY not async: called from __main__.py before uvicorn starts.
    The network call is blocking, but it's a one-time startup cost.
    Returns a status dict for the startup banner.
    """
    key = get_license_key()

    if not key:
        return {
            "tier": "free",
            "patterns": len(load_patterns()),
            "message": "Free tier — 100 patterns. Run 'judgement activate <KEY>' to unlock Elite.",
        }

    result = validate_license(key)

    if not result.get("valid"):
        error = result.get("error", "Unknown error")
        return {
            "tier": "free",
            "patterns": len(load_patterns()),
            "message": f"License invalid: {error}. Falling back to free tier (100 patterns).",
        }

    tier = result.get("tier", "free")
    sync = sync_patterns(key)

    if sync.get("success"):
        pattern_count = sync["count"]
        mt_count = sync.get("multi_turn_count", 0)
        msg = f"Elite {tier.replace('elite_', '').title()} — {pattern_count:,} patterns synced."
        if mt_count:
            msg += f" {mt_count} multi-turn patterns loaded."
        return {"tier": tier, "patterns": pattern_count, "multi_turn": mt_count, "message": msg}
    else:
        cached = load_patterns()
        return {
            "tier": tier,
            "patterns": len(cached),
            "message": f"Pattern sync failed ({sync.get('error')}). Using {len(cached)} cached patterns.",
        }


# === SECTION: CLI COMMANDS === #

def activate(key: str) -> bool:
    """Activate a license key (CLI: `judgement activate <KEY>`)."""
    key = key.strip().upper()

    if not key.startswith("FAS-") or len(key) != 23:
        print("❌ Invalid key format. Expected: FAS-XXXX-XXXX-XXXX-XXXX")
        return False

    print(f"Activating license: {key[:8]}{'*' * 15}")
    print(f"Machine ID: {get_machine_id()}")
    print()

    result = validate_license(key)
    if result.get("valid"):
        tier = result["tier"]
        features = result.get("features", {})
        print("✅ License activated!")
        print(f"   Tier: {tier.replace('_', ' ').title()}")
        print(f"   Multi-turn: {'Yes' if features.get('multi_turn') else 'No'}")
        print(f"   Teams: {'Yes' if features.get('teams') else 'No'}")
        if result.get("expires_at"):
            print(f"   Expires: {result['expires_at'][:10]}")
        if result.get("seats"):
            print(f"   Seats: {result.get('seats_used', 0)}/{result['seats']}")
        print()

        print("Syncing patterns...")
        sync = sync_patterns(key)
        if sync.get("success"):
            print(f"✅ {sync['count']:,} patterns downloaded and cached.")
            if sync.get("multi_turn_count"):
                print(f"   {sync['multi_turn_count']} multi-turn patterns included.")
        else:
            print(f"⚠️  Pattern sync failed: {sync.get('error')}")
            print("   Patterns will sync on next startup.")

        return True
    else:
        print(f"❌ Activation failed: {result.get('error', 'Unknown error')}")
        return False


def deactivate() -> None:
    """Deactivate the current license (CLI: `judgement deactivate`)."""
    config = load_config()
    if not config.get("license_key"):
        print("No active license found.")
        return
    clear_config()
    print("✅ License deactivated. Reverted to free tier (100 patterns).")


def status() -> None:
    """Show current license status (CLI: `judgement status`)."""
    config = load_config()
    key = config.get("license_key")

    if not key:
        print("Tier: Free")
        print(f"Patterns: {len(load_patterns())}")
        print("\nRun 'judgement activate <KEY>' to upgrade to Elite.")
        return

    print(f"License: {key[:8]}{'*' * 15}")
    print(f"Tier: {config.get('tier', 'unknown').replace('_', ' ').title()}")
    print(f"Last validated: {config.get('last_validated', 'never')}")
    if config.get("expires_at"):
        print(f"Expires: {config['expires_at'][:10]}")

    features = config.get("features", {})
    print("\nFeatures:")
    print(f"  Multi-turn: {'✅' if features.get('multi_turn') else '❌'}")
    print(f"  Teams: {'✅' if features.get('teams') else '❌'}")
    print(f"  Campaigns: {'✅' if features.get('campaigns') else '❌'}")
    print(f"  Reports: {'✅' if features.get('reports') else '❌'}")

    patterns = load_patterns()
    mt = load_multi_turn_patterns()
    print(f"\nPatterns: {len(patterns):,}")
    if mt:
        print(f"Multi-turn: {len(mt)}")

    if CACHED_PATTERNS_FILE.exists():
        try:
            cache = json.loads(CACHED_PATTERNS_FILE.read_text())
            print(f"Last synced: {cache.get('synced_at', 'unknown')}")
            print(f"Version: {cache.get('version', 'unknown')}")
        except Exception:
            pass
