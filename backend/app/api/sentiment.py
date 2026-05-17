from fastapi import APIRouter, Query, Path
from typing import Optional
from backend.app.services.sentiment_service import get_sentiment_by_topic, get_overall_sentiment

router = APIRouter()


@router.get("/overall")
async def get_overall_sentiment_endpoint(
    platform: Optional[str] = Query(None),
    hours: int = Query(24),
):
    results = await get_overall_sentiment(platform=platform, hours=hours)
    return {"sentiment_breakdown": results, "window_hours": hours}


@router.get("/{topic}")
async def get_topic_sentiment(
    topic: str = Path(...),
    hours: int = Query(24),
):
    results = await get_sentiment_by_topic(topic=topic, hours=hours)
    return {"topic": topic, "timeline": results, "window_hours": hours}
