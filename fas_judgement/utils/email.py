"""
Email Utilities (OSS Edition)
-------------------------------
WHY: OSS doesn't send emails — no Stripe purchases, no license emails,
     no user accounts. This stub exists so any code that tries to import
     from utils.email doesn't break.

     Elite version sends license keys via Resend. That lives in the hosted
     build, not here.

LAYER: Utils (infrastructure) — no imports needed.
"""


def send_license_email(
    to_email: str,
    customer_name: str,
    license_key: str,
    tier: str,
) -> None:
    """
    No-op stub. OSS doesn't send emails.
    WHY stub (not deleted): keeps Elite/OSS import paths consistent
    so shared code doesn't need conditional imports.
    """
    pass
