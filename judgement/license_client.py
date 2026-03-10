"""
FAS Judgement - License Client

Handles license activation, validation, and pattern syncing for the local app.
Communicates with the FAS License API on the Judgement VPS.
"""

import hashlib
import json
import os
import platform
import sys
import uuid
from datetime import datetime
from pathlib import Path

import httpx

# API base URL
FAS_API_URL = os.environ.get(
    "FAS_API_URL", "https://judgement-app.fallenangelsystems.com"
)

# Local config directory
CONFIG_DIR = Path.home() / ".judgement"
CONFIG_FILE = CONFIG_DIR / "config.json"
CACHED_PATTERNS_FILE = CONFIG_DIR / "patterns_cache.json"
BUNDLED_PATTERNS = Path(__file__).parent / "patterns.json"

# Number of free patterns (bundled with OSS)
FREE_PATTERN_COUNT = 100


def get_machine_id() -> str:
    """Generate a stable machine identifier."""
    parts = [
        platform.node(),
        platform.machine(),
        platform.system(),
        str(uuid.getnode()),  # MAC address
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_config() -> dict:
    """Load saved config (license key, tier, etc)."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_config(config: dict):
    """Save config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def clear_config():
    """Clear saved config and cached patterns."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
    if CACHED_PATTERNS_FILE.exists():
        CACHED_PATTERNS_FILE.unlink()


def get_license_key() -> str | None:
    """Get the license key from env var or saved config."""
    # Env var takes priority
    key = os.environ.get("FAS_LICENSE_KEY", "").strip()
    if key:
        return key.upper()

    # Fall back to saved config
    config = load_config()
    return config.get("license_key")


def validate_license(key: str = None) -> dict:
    """
    Validate license key against FAS API.
    Returns validation result dict with tier, features, etc.
    On failure, returns {"valid": False, "error": "..."}.
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
                # Save validation result to config
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
                # Clear saved config on invalid/revoked key
                clear_config()
                return {"valid": False, "error": data.get("error", "License invalid"), "tier": "free"}
            else:
                return {"valid": False, "error": f"API error ({resp.status_code})", "tier": "free"}

    except httpx.ConnectError:
        return {"valid": False, "error": "Cannot reach FAS license server", "tier": "free"}
    except httpx.TimeoutException:
        return {"valid": False, "error": "License server timeout", "tier": "free"}
    except Exception as e:
        return {"valid": False, "error": str(e), "tier": "free"}


def sync_patterns(key: str = None) -> dict:
    """
    Download the full pattern library from FAS API.
    Caches patterns locally for performance.
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

                # Cache patterns locally
                CONFIG_DIR.mkdir(parents=True, exist_ok=True)
                cache_data = {
                    "patterns": patterns,
                    "synced_at": datetime.utcnow().isoformat(),
                    "version": data.get("version", "unknown"),
                    "count": len(patterns),
                }

                # Include multi-turn if present
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


def load_patterns() -> list:
    """
    Load patterns based on license status.
    Priority: cached Elite patterns > bundled free patterns.
    """
    config = load_config()
    tier = config.get("tier", "free")

    # If we have a valid Elite license and cached patterns, use those
    if tier in ("elite_home", "elite_business") and CACHED_PATTERNS_FILE.exists():
        try:
            cache = json.loads(CACHED_PATTERNS_FILE.read_text())
            patterns = cache.get("patterns", [])
            if patterns:
                return patterns
        except (json.JSONDecodeError, OSError):
            pass

    # Fall back to bundled free patterns
    if BUNDLED_PATTERNS.exists():
        try:
            all_patterns = json.loads(BUNDLED_PATTERNS.read_text())
            # Free tier only gets first 100
            return all_patterns[:FREE_PATTERN_COUNT]
        except (json.JSONDecodeError, OSError):
            pass

    return []


def load_multi_turn_patterns() -> list:
    """Load multi-turn patterns (Elite only)."""
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
    """Get the current license tier."""
    config = load_config()
    return config.get("tier", "free")


def get_features() -> dict:
    """Get current feature flags."""
    config = load_config()
    return config.get("features", {
        "patterns": True,
        "multi_turn": False,
        "teams": False,
        "campaigns": False,
        "reports": False,
        "custom_patterns": False,
        "max_users": 1,
    })


def is_feature_enabled(feature: str) -> bool:
    """Check if a specific feature is enabled for current tier."""
    features = get_features()
    return features.get(feature, False)


def startup_check() -> dict:
    """
    Run on app startup. Validates license and syncs patterns if needed.
    Returns status dict for display.
    """
    key = get_license_key()

    if not key:
        return {
            "tier": "free",
            "patterns": len(load_patterns()),
            "message": "Free tier - 100 patterns. Run 'judgement activate <KEY>' to upgrade.",
        }

    # Validate license
    result = validate_license(key)

    if not result.get("valid"):
        error = result.get("error", "Unknown error")
        return {
            "tier": "free",
            "patterns": len(load_patterns()),
            "message": f"License invalid: {error}. Falling back to free tier (100 patterns).",
        }

    tier = result.get("tier", "free")

    # Sync patterns
    sync = sync_patterns(key)
    if sync.get("success"):
        pattern_count = sync["count"]
        mt_count = sync.get("multi_turn_count", 0)
        msg = f"Elite {tier.replace('elite_', '').title()} - {pattern_count:,} patterns synced."
        if mt_count:
            msg += f" {mt_count} multi-turn patterns loaded."
        return {
            "tier": tier,
            "patterns": pattern_count,
            "multi_turn": mt_count,
            "message": msg,
        }
    else:
        # Sync failed but license is valid - try cached patterns
        cached = load_patterns()
        return {
            "tier": tier,
            "patterns": len(cached),
            "message": f"Pattern sync failed ({sync.get('error')}). Using {len(cached)} cached patterns.",
        }


# ─── CLI Commands ─────────────────────────────────────────────

def activate(key: str):
    """Activate a license key."""
    key = key.strip().upper()

    if not key.startswith("FAS-") or len(key) != 23:
        print("❌ Invalid key format. Expected: FAS-XXXX-XXXX-XXXX-XXXX")
        return False

    print(f"Activating license: {key[:8]}{'*' * 15}")
    print(f"Machine ID: {get_machine_id()}")
    print()

    # Validate
    result = validate_license(key)

    if result.get("valid"):
        tier = result["tier"]
        features = result.get("features", {})
        print(f"✅ License activated!")
        print(f"   Tier: {tier.replace('_', ' ').title()}")
        print(f"   Multi-turn: {'Yes' if features.get('multi_turn') else 'No'}")
        print(f"   Teams: {'Yes' if features.get('teams') else 'No'}")
        if result.get("expires_at"):
            print(f"   Expires: {result['expires_at'][:10]}")
        if result.get("seats"):
            print(f"   Seats: {result.get('seats_used', 0)}/{result['seats']}")
        print()

        # Sync patterns
        print("Syncing patterns...")
        sync = sync_patterns(key)
        if sync.get("success"):
            print(f"✅ {sync['count']:,} patterns downloaded and cached.")
            if sync.get("multi_turn_count"):
                print(f"   {sync['multi_turn_count']} multi-turn patterns included.")
        else:
            print(f"⚠️  Pattern sync failed: {sync.get('error')}")
            print("   Patterns will sync on next app startup.")

        return True
    else:
        error = result.get("error", "Unknown error")
        print(f"❌ Activation failed: {error}")
        return False


def deactivate():
    """Deactivate the current license."""
    config = load_config()
    if not config.get("license_key"):
        print("No active license found.")
        return

    clear_config()
    print("✅ License deactivated. Reverted to free tier (100 patterns).")


def status():
    """Show current license status."""
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
    print(f"\nFeatures:")
    print(f"  Multi-turn: {'✅' if features.get('multi_turn') else '❌'}")
    print(f"  Teams: {'✅' if features.get('teams') else '❌'}")
    print(f"  Campaigns: {'✅' if features.get('campaigns') else '❌'}")
    print(f"  Reports: {'✅' if features.get('reports') else '❌'}")

    # Pattern counts
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
