from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import httpx

from dependencies import get_current_user_id
from config import settings

router = APIRouter()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    segmentation: Optional[Any] = None


SYSTEM_INSTRUCTION = (
    "[CONTEXT FOR AI — not shown to user] "
    "You are OralSkop's dental AI assistant. The patient is already using OralSkop, "
    "an app that automatically handles dentist referral and appointment scheduling. "
    "NEVER suggest, recommend, or mention booking a dental appointment, visiting a dentist, "
    "scheduling a check-up, or seeking professional care — the app already does this. "
    "Focus exclusively on: explaining what the detected conditions mean, how serious they are, "
    "what causes them, and general oral health education. "
    "Be clear, empathetic, and concise. Use plain language, not clinical jargon. "
    "[END CONTEXT]\n\n"
)


@router.post("/chat")
async def chat(
    body: ChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    ai_url = settings.ai_server_url.rstrip("/") + "/chat"

    # Inject system instruction as prefix on the first user message
    messages = [m.model_dump() for m in body.messages]
    if messages and messages[0]["role"] == "user":
        messages[0] = {
            "role": "user",
            "content": SYSTEM_INSTRUCTION + messages[0]["content"],
        }

    payload = {
        "messages": messages,
        "segmentation": body.segmentation,
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                ai_url,
                json=payload,
                headers={"ngrok-skip-browser-warning": "true"},
            )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"AI server unreachable: {e}")
