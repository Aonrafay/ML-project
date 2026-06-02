import httpx
from backend.app.core.config import get_settings


def google_fact_check(claim: str) -> dict:
    settings = get_settings()
    if not settings.google_fact_check_api_key:
        return {"error": "Google Fact Check API key not configured"}

    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    params = {
        "query": claim,
        "key": settings.google_fact_check_api_key,
        "languageCode": "en",
    }

    try:
        with httpx.Client() as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        claims = data.get("claims", [])
        if not claims:
            return {"verdict": "no_match", "claims": [], "confidence": 0}

        results = []
        for c in claims[:5]:
            claim_review = c.get("claimReview", [{}])[0]
            results.append({
                "claim_text": c.get("text", ""),
                "verdict": claim_review.get("textualRating", "unknown"),
                "publisher": claim_review.get("publisher", {}).get("name", "unknown"),
                "url": claim_review.get("url", ""),
                "date": claim_review.get("reviewDate", ""),
            })

        return {
            "verdict": results[0]["verdict"] if results else "unknown",
            "claims": results,
            "confidence": 0.8 if results else 0,
        }

    except Exception as e:
        return {"error": str(e), "verdict": "error", "confidence": 0}
