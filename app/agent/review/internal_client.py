import os

def get_internal_headers() -> dict[str, str]:
    api_key = os.getenv("INTERNAL_API_KEY", "")
    if not api_key:
        raise RuntimeError("INTERNAL_API_KEY is not configured")

    return {"X-Internal-Api-Key": api_key}