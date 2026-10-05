"""Health check script for monitoring."""
from __future__ import annotations

import sys
import urllib.request


def check_health(url: str = "http://localhost:8501") -> bool:
    """Return True if the app is responding."""
    try:
        response = urllib.request.urlopen(url, timeout=5)
        return response.status == 200
    except Exception:
        return False


if __name__ == "__main__":
    healthy = check_health()
    print("OK" if healthy else "UNHEALTHY")
    sys.exit(0 if healthy else 1)
