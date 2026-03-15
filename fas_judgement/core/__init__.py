"""
Core Domain Layer
-----------------
WHY: The core package contains the pure domain — data structures, enums,
     interfaces, and errors that have ZERO dependency on HTTP, databases,
     or external services.

DDD principle: the domain is the innermost ring. It knows nothing about
FastAPI, SQLite, Stripe, or OAuth. That lets us test business logic without
spinning up a web server or touching the filesystem.
"""
