from fastapi import APIRouter, Depends, Query

from dependencies import get_current_user_id
from services.db import get_db
from services.s3_service import get_presigned_url
from config import settings

router = APIRouter()


@router.get("/history")
async def get_history(
    user_id: str = Depends(get_current_user_id),
    limit: int = Query(default=5, ge=1, le=50),
):
    db = get_db()

    screenings_resp = (
        db.table("screenings")
        .select("screening_id, captured_at, photo_url, masked_image_url, has_escalation")
        .eq("user_id", user_id)
        .order("captured_at", desc=True)
        .limit(limit)
        .execute()
    )

    screenings = screenings_resp.data or []
    if not screenings:
        return []

    # Batch-fetch all detections for these screenings in a single query
    screening_ids = [s["screening_id"] for s in screenings]
    dets_resp = (
        db.table("detections")
        .select("screening_id, condition_type")
        .in_("screening_id", screening_ids)
        .execute()
    )
    # Group by screening_id
    dets_by_screening: dict = {}
    for d in (dets_resp.data or []):
        dets_by_screening.setdefault(d["screening_id"], []).append(d["condition_type"])

    result = []
    for s in screenings:
        masked_url = get_presigned_url(settings.s3_bucket_masked, s["masked_image_url"])
        photo_url  = get_presigned_url(settings.s3_bucket_photos, s["photo_url"])

        result.append({
            "screening_id": s["screening_id"],
            "captured_at": s["captured_at"],
            "photo_url": photo_url,
            "masked_image_url": masked_url,
            "escalation_triggered": s["has_escalation"],
            "condition_summary": dets_by_screening.get(s["screening_id"], []),
        })

    return result
