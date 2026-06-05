from fastapi import APIRouter, Depends

from dependencies import get_current_user_id
from services.db import get_db
from services.s3_service import get_presigned_url
from config import settings

router = APIRouter()


@router.get("/history")
async def get_history(user_id: str = Depends(get_current_user_id)):
    db = get_db()

    screenings_resp = (
        db.table("screenings")
        .select("screening_id, captured_at, masked_image_url, has_escalation")
        .eq("user_id", user_id)
        .order("captured_at", desc=True)
        .execute()
    )

    screenings = screenings_resp.data or []

    # Fetch condition summaries and attach presigned URLs
    result = []
    for s in screenings:
        dets_resp = (
            db.table("detections")
            .select("condition_type")
            .eq("screening_id", s["screening_id"])
            .execute()
        )
        condition_summary = [d["condition_type"] for d in (dets_resp.data or [])]

        masked_url = get_presigned_url(settings.s3_bucket_masked, s["masked_image_url"])

        result.append({
            "screening_id": s["screening_id"],
            "captured_at": s["captured_at"],
            "masked_image_url": masked_url,
            "escalation_triggered": s["has_escalation"],
            "condition_summary": condition_summary,
        })

    return result
