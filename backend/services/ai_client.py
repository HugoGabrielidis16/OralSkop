import base64
import httpx
from config import settings
from mock.mock_response import MOCK_AI_RESPONSE

# Hardcoded recommendation bullets per condition
RECOMMENDATIONS: dict[str, list[str]] = {
    "cavity": [
        "Limit sugary and acidic foods",
        "Brush twice daily with fluoride toothpaste",
        "Consult your dentist within 1 month",
    ],
    "gingivitis": [
        "Gentle brushing 2×/day along the gumline",
        "Use antiseptic mouthwash for a few days",
        "Have it checked by a dentist within 3–4 weeks",
    ],
    "tartar": [
        "Tartar cannot be removed by brushing alone",
        "Schedule a professional cleaning with your dentist",
    ],
    "lesion_suspicious": [
        "This area requires professional evaluation",
        "Please consult an oral health specialist or doctor for orientation",
    ],
}


def _confidence_to_severity(score: float) -> str:
    if score >= 0.85:
        return "high"
    if score >= 0.65:
        return "moderate"
    return "low"


def _parse_predictions(predictions: list[dict]) -> tuple[list[dict], bool]:
    """
    Enrich raw AI predictions with severity, tooth_number (mocked null),
    and recommendations. Also returns escalation flag.
    """
    detections = []
    escalation = False

    for p in predictions:
        condition = p["condition"]
        confidence = p["confidence"]

        if condition == "lesion_suspicious":
            escalation = True

        detections.append(
            {
                "condition": condition,
                "confidence": confidence,
                "severity": _confidence_to_severity(confidence),
                "tooth_number": None,  # mocked until AI mapping module delivered
                "box_coordinates": p.get("box_coordinates"),
                "recommendations": RECOMMENDATIONS.get(condition, []),
            }
        )

    return detections, escalation


async def analyze_image(file_bytes: bytes) -> dict:
    """
    Main entrypoint. Returns enriched analysis dict.
    USE_MOCK_AI=true → returns mock fixture instantly.
    USE_MOCK_AI=false → calls real AI server.
    """
    if settings.use_mock_ai:
        raw = MOCK_AI_RESPONSE
    else:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                settings.ai_server_url,
                files={"file": ("photo.jpg", file_bytes, "image/jpeg")},
            )
            response.raise_for_status()
            raw = response.json()

    detections, escalation = _parse_predictions(raw["predictions"])
    masked_image_b64: str = raw["masked_image"]
    masked_image_bytes: bytes = base64.b64decode(masked_image_b64)

    return {
        "masked_image_bytes": masked_image_bytes,
        "detections": detections,
        "escalation_triggered": escalation,
    }
