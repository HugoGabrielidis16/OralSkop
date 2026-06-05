"""
Hardcoded mock AI response.
masked_image is a 1x1 transparent PNG encoded as base64 — replace with a
realistic pre-masked mouth photo before the demo.
"""

MOCK_MASKED_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

MOCK_AI_RESPONSE = {
    "status": "success",
    "masked_image": MOCK_MASKED_B64,
    "predictions": [
        {
            "condition": "gingivitis",
            "confidence": 0.86,
            "box_coordinates": [120, 45, 200, 180],
        },
        {
            "condition": "tartar",
            "confidence": 0.74,
            "box_coordinates": [300, 60, 380, 160],
        },
    ],
}
