import uuid
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

from dependencies import get_current_user_id
from services.ai_client import analyze_image
from services.s3_service import upload_bytes, get_presigned_url
from services.db import get_db
from config import settings

router = APIRouter()


@router.post("/screenings", status_code=201)
async def create_screening(
    photo: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    file_bytes = await photo.read()
    screening_id = str(uuid.uuid4())

    # 1. Upload raw photo to S3
    photo_key = f"photos/{user_id}/{screening_id}.jpg"
    upload_bytes(settings.s3_bucket_photos, photo_key, file_bytes, "image/jpeg")

    # 2. Run AI analysis (mock or real)
    try:
        result = await analyze_image(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI inference failed: {str(e)}")

    # 3. Upload masked image to S3
    masked_key = f"masked/{user_id}/{screening_id}_masked.png"
    upload_bytes(settings.s3_bucket_masked, masked_key, result["masked_image_bytes"], "image/png")

    # 4. Generate presigned URLs
    photo_url = get_presigned_url(settings.s3_bucket_photos, photo_key)
    masked_image_url = get_presigned_url(settings.s3_bucket_masked, masked_key)

    # 5. Persist to DB
    db = get_db()
    # Upsert user row — creates it on first screening if it doesn't exist yet.
    # Covers both email/password signups and future OAuth logins.
    db.table("users").upsert({"user_id": user_id}, on_conflict="user_id").execute()

    db.table("screenings").insert({
        "screening_id": screening_id,
        "user_id": user_id,
        "photo_url": photo_key,          # store S3 key, not the expiring URL
        "masked_image_url": masked_key,  # store S3 key
        "has_escalation": result["escalation_triggered"],
    }).execute()

    for det in result["detections"]:
        db.table("detections").insert({
            "screening_id": screening_id,
            "condition_type": det["condition"],
            "confidence_score": det["confidence"],
            "arcade_tooth_number": det["tooth_number"],
            "box_coordinates": det["box_coordinates"],
        }).execute()

    # 6. Return response with presigned URLs
    return {
        "screening_id": screening_id,
        "photo_url": photo_url,
        "masked_image_url": masked_image_url,
        "escalation_triggered": result["escalation_triggered"],
        "detections": [
            {
                "condition": d["condition"],
                "confidence": d["confidence"],
                "severity": d["severity"],
                "tooth_number": d["tooth_number"],
                "recommendations": d["recommendations"],
            }
            for d in result["detections"]
        ],
    }
