import httpx
from backend.app.core.config import get_settings


def claimbuster_check(text: str) -> dict:
    settings = get_settings()
    if not settings.claimbuster_api_key:
        return {"error": "ClaimBuster API key not configured"}

    url = "https://api.idaholab.ai/claimbuster/v1/score"
    params = {
        "text": text,
        "key": settings.claimbuster_api_key,
    }

    try:
        with httpx.Client() as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        return {
            "score": data.get("score", 0),
            "checkworthy": data.get("score", 0) > 0.5,
            "raw": data,
        }

    except Exception as e:
        return {"error": str(e), "score": 0, "checkworthy": False}
