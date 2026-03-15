"""
Core Domain Errors
------------------
WHY: A typed error hierarchy lets the HTTP layer map domain failures to
     appropriate HTTP status codes without the domain knowing about HTTP.

     Pattern: raise DomainError in services → http layer catches and
     converts to HTTPException or JSONResponse. This keeps business rules
     (e.g. "pro tier required") separate from protocol concerns (HTTP 403).

LAYER: Core domain — no framework imports.
"""


# === SECTION: BASE === #

class FASError(Exception):
    """Root exception for all FAS Judgement domain errors."""
    pass


# === SECTION: AUTH / ACCESS ERRORS === #

class AuthError(FASError):
    """Authentication failed — missing or invalid credentials."""
    pass


class PermissionError(FASError):
    """
    Authenticated user lacks permission for this action.
    Different from AuthError: the user IS known, they just can't do this.
    """
    pass


class TierError(PermissionError):
    """
    Feature requires a higher subscription tier.
    Carries `required_tier` so the HTTP layer can return a helpful upgrade message.
    """
    def __init__(self, message: str, required_tier: str = "pro"):
        super().__init__(message)
        self.required_tier = required_tier


# === SECTION: RESOURCE ERRORS === #

class NotFoundError(FASError):
    """Requested resource does not exist."""
    pass


class DuplicateError(FASError):
    """Resource already exists (e.g. duplicate pattern submission)."""
    pass


# === SECTION: VALIDATION ERRORS === #

class ValidationError(FASError):
    """Input data failed domain-level validation (beyond pydantic schema)."""
    pass


class SSRFError(ValidationError):
    """
    URL points to a private/internal address — blocked by SSRF protection.
    WHY: We raise this specifically so the HTTP layer can return a clear
    message rather than a generic 400.
    """
    pass


# === SECTION: RATE LIMIT === #

class RateLimitError(FASError):
    """
    User has exceeded the allowed request rate.
    Carries `retry_after_seconds` for the Retry-After header.
    """
    def __init__(self, message: str, retry_after_seconds: int = 60):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


# === SECTION: EXTERNAL SERVICE ERRORS === #

class LicenseError(FASError):
    """License validation failed."""
    pass


class TransportError(FASError):
    """Could not reach or communicate with an attack target."""
    pass
