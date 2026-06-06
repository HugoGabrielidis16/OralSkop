import base64
import io
import httpx
from config import settings
from mock.mock_response import MOCK_AI_RESPONSE


def _to_jpeg(file_bytes: bytes, content_type: str | None) -> tuple[bytes, str]:
    """
    If the image is HEIC/HEIF (iOS camera default), convert to JPEG.
    Returns (jpeg_bytes, 'image/jpeg'). All other formats pass through.
    """
    mime = content_type or ''
    is_heic = (
        mime.lower() in ('image/heic', 'image/heif')
        or file_bytes[4:12] in (b'ftypheic', b'ftypheix', b'ftyphevc', b'ftyphevx', b'ftypmif1', b'ftypmsf1')
    )
    if not is_heic:
        return file_bytes, mime or _detect_mime(file_bytes)

    import pillow_heif
    from PIL import Image
    pillow_heif.register_heif_opener()
    img = Image.open(io.BytesIO(file_bytes))
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='JPEG', quality=92)
    return buf.getvalue(), 'image/jpeg'

# Hardcoded recommendation bullets per condition
RECOMMENDATIONS: dict[str, list[str]] = {
    "cavity": [
        "Limit sugary and acidic foods",
        "Brush twice daily with fluoride toothpaste",
        "Consult your dentist within 1 month",
    ],
    "caries": [
        "Limit sugary and acidic foods",
        "Brush twice daily with fluoride toothpaste",
        "Consult your dentist within 1 month",
    ],
    "abrasion": [
        "Avoid brushing with excessive pressure",
        "Switch to a soft-bristle toothbrush",
        "Ask your dentist about a desensitising toothpaste",
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
    "crown": [
        "An existing crown was detected — monitor for chips or sensitivity",
        "Mention it at your next dental check-up",
    ],
}

# Map real AI class_name values to our internal condition names
CONDITION_MAP: dict[str, str] = {
    "caries": "caries",
    "abrasion": "abrasion",
    "cavity": "cavity",
    "gingivitis": "gingivitis",
    "tartar": "tartar",
    "lesion_suspicious": "lesion_suspicious",
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

    Accepts both our internal format (condition + box_coordinates)
    and the real AI format (class_name + bbox).
    """
    detections = []
    escalation = False

    for p in predictions:
        # Support real AI field names (class_name, bbox) and our internal names
        condition = CONDITION_MAP.get(
            p.get("class_name", p.get("condition", "unknown")),
            p.get("class_name", p.get("condition", "unknown")),
        )
        confidence = p["confidence"]
        box = p.get("bbox") or p.get("box_coordinates")

        if condition == "lesion_suspicious":
            escalation = True

        detections.append(
            {
                "condition": condition,
                "confidence": confidence,
                "severity": _confidence_to_severity(confidence),
                "tooth_number": None,
                "box_coordinates": box,
                "recommendations": RECOMMENDATIONS.get(condition, []),
            }
        )

    return detections, escalation


def _detect_mime(file_bytes: bytes) -> str:
    """Detect image MIME type from magic bytes."""
    if file_bytes[:2] == b'\xff\xd8':
        return 'image/jpeg'
    if file_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return 'image/png'
    if file_bytes[:4] == b'RIFF' and file_bytes[8:12] == b'WEBP':
        return 'image/webp'
    return 'image/jpeg'  # fallback


async def analyze_image(file_bytes: bytes, content_type: str | None = None) -> dict:
    """
    Main entrypoint. Returns enriched analysis dict.
    USE_MOCK_AI=true → returns mock fixture instantly.
    USE_MOCK_AI=false → calls real AI server.
    """
    # Convert HEIC→JPEG before anything else
    file_bytes, content_type = _to_jpeg(file_bytes, content_type)
    if settings.use_mock_ai:
        detections, escalation = _parse_predictions(MOCK_AI_RESPONSE["predictions"])
        masked_image_bytes = base64.b64decode(MOCK_AI_RESPONSE["masked_image"])
    else:
        base_url = settings.ai_server_url.rstrip("/")
        mime = content_type or _detect_mime(file_bytes)
        ext = mime.split('/')[-1].replace('jpeg', 'jpg')
        filename = f"photo.{ext}"
        headers = {"ngrok-skip-browser-warning": "true"}
        async with httpx.AsyncClient(timeout=60) as client:
            json_resp = await client.post(
                f"{base_url}/predict",
                files={"file": (filename, file_bytes, mime)},
                headers=headers,
            )
            json_resp.raise_for_status()
            overlay_resp = await client.post(
                f"{base_url}/predict/overlay",
                files={"file": (filename, file_bytes, mime)},
                headers=headers,
            )
            overlay_resp.raise_for_status()

        raw = json_resp.json()
        masked_image_bytes = overlay_resp.content  # PNG bytes directly
        detections, escalation = _parse_predictions(raw["detections"])

    return {
        "masked_image_bytes": masked_image_bytes,
        "detections": detections,
        "escalation_triggered": escalation,
    }
