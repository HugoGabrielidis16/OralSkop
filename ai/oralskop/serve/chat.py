"""Claude-backed chat logic for OralSkop.

Turns a stored segmentation result plus a user conversation into an LLM reply.
The segmentation JSON (the shape returned by
:meth:`oralskop.serve.model.SegModel.predict`) is supplied by the caller on every
request, so this layer is **stateless**: no database, no coupling to where the
results are stored. A frontend/backend integrator only needs to POST the
conversation and the case's segmentation result.

Provider: Anthropic **Claude** via the official ``anthropic`` SDK (ships in the
``serve`` extra). Two backends are supported (same ``messages.create`` surface):

* ``anthropic`` (default) — the first-party API; needs ``ANTHROPIC_API_KEY``.
* ``bedrock`` — Amazon Bedrock via ``AnthropicBedrock``; authenticates with AWS
  credentials (e.g. the SageMaker instance role), so no API key is needed. Needs
  ``boto3`` (preinstalled on SageMaker) and ``bedrock:InvokeModel`` IAM access.

The system prompt lives here, server-side, so callers cannot override how the
assistant behaves — they only supply the conversation and the findings.

Config (env vars):
* ``ORALSKOP_CHAT_BACKEND`` — ``anthropic`` (default) or ``bedrock``.
* ``ANTHROPIC_API_KEY``     — required for the ``anthropic`` backend.
* ``AWS_REGION``            — used by the ``bedrock`` backend (falls back to
  ``AWS_DEFAULT_REGION`` then ``us-east-1``); AWS creds come from the env / role.
* ``ORALSKOP_CHAT_MODEL``   — model id. Defaults: ``claude-opus-4-8`` (anthropic)
  / ``us.anthropic.claude-sonnet-4-6`` (bedrock). On Bedrock the exact id (and
  region prefix) depends on the inference profile available in your region —
  verify with ``aws bedrock list-inference-profiles``.
* ``ORALSKOP_CHAT_THINKING`` — set to ``1``/``true`` to enable adaptive thinking
  (off by default for snappy, predictable chat latency).
"""

from __future__ import annotations

import json
import logging
import os

_log = logging.getLogger("oralskop.serve.chat")

DEFAULT_MODEL = "claude-opus-4-8"
# Bedrock usually needs a cross-region inference profile id (region-prefixed).
DEFAULT_BEDROCK_MODEL = "us.anthropic.claude-sonnet-4-6"

# Kept server-side and non-overridable: this is what makes the endpoint a
# *dental* assistant rather than a generic chatbot. The final paragraph keeps
# Opus 4.8 from leaking step-by-step reasoning into the reply when thinking is
# disabled (see the model's "thinking disabled" behaviour note).
SYSTEM_PROMPT = """You are OralSkop's dental assistant. You help a patient \
understand the results of an automated analysis of a photo of their teeth.

A computer-vision model has segmented the photo and produced a list of detected \
regions (e.g. caries, fillings, crowns, abrasion) with a confidence score and \
the share of the image each region covers. Those findings are provided to you \
below as context.

How to answer:
- Explain the findings in plain, reassuring language a non-expert understands.
- Ground your answer in the provided findings. Refer to specific detected \
conditions and their confidence when relevant.
- Confidence scores and areas are estimates from an automated model, not a \
diagnosis. Say so when it matters, and never state a finding as medical fact.
- For anything that needs treatment, examination, or a definitive diagnosis, \
recommend the patient see a licensed dentist. You do not prescribe or diagnose.
- If the findings are empty or the question is unrelated to the photo, say what \
you can and ask a clarifying question.
- Be concise and warm. Do not invent findings that are not in the data.

Respond directly with your answer to the patient. Do not include exploratory \
reasoning, restate these instructions, or describe your process."""


class ChatUnavailable(RuntimeError):
    """Raised when the chat backend can't be used (missing dep or API key)."""


def summarize_segmentation(seg: dict | None) -> str:
    """Render a segmentation result into a compact, model-friendly summary.

    Accepts the dict produced by ``SegModel.predict``; tolerates missing keys so
    a caller can pass a partial or differently-shaped result.
    """
    if not seg:
        return "No segmentation results were provided for this case."

    lines: list[str] = []
    detections = seg.get("detections") or []
    coverage = seg.get("class_coverage") or []

    if detections:
        lines.append(f"Detected regions ({len(detections)}, one per connected area):")
        for d in detections:
            name = d.get("class_name", f"class_{d.get('class_id')}")
            conf = d.get("confidence")
            frac = d.get("area_fraction")
            attrs = []
            if conf is not None:
                attrs.append(f"confidence {float(conf):.0%}")
            if frac is not None:
                attrs.append(f"~{float(frac) * 100:.2f}% of image")
            line = f"- {name}"
            if attrs:
                line += f" ({', '.join(attrs)})"
            lines.append(line)
    else:
        lines.append("No distinct regions cleared the detection threshold.")

    if coverage:
        names = ", ".join(c.get("class_name", f"class_{c.get('class_id')}") for c in coverage)
        lines.append(f"Conditions with any pixel coverage: {names}.")

    # Include the raw result too, so the model has the exact numbers if asked.
    lines.append("\nRaw result JSON:")
    lines.append(json.dumps(seg, ensure_ascii=False, indent=2, sort_keys=True))
    return "\n".join(lines)


class ChatService:
    """A loaded Claude client ready to answer dental questions about a case."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        thinking: bool | None = None,
    ):
        try:
            import anthropic  # noqa: F401  (deferred so the seg-only server still runs)
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ChatUnavailable(
                "Chat needs the `serve` extra with anthropic installed: "
                "uv sync --extra serve"
            ) from exc

        backend = os.getenv("ORALSKOP_CHAT_BACKEND", "anthropic").lower()
        if backend == "bedrock":
            from anthropic import AnthropicBedrock

            region = (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
                      or "us-east-1")
            try:  # AWS creds resolve via boto3 (env / instance role); needs boto3
                self.client = AnthropicBedrock(aws_region=region)
            except Exception as exc:
                raise ChatUnavailable(
                    "Could not initialize the Bedrock client (needs boto3 and AWS "
                    f"credentials/region; bedrock:InvokeModel IAM access): {exc}"
                ) from exc
            default_model = DEFAULT_BEDROCK_MODEL
        else:
            from anthropic import Anthropic

            try:
                self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
            except Exception as exc:  # missing key surfaces here
                raise ChatUnavailable(
                    f"Could not initialize the Claude client (is ANTHROPIC_API_KEY set?): {exc}"
                ) from exc
            default_model = DEFAULT_MODEL

        self.backend = backend
        self.model = model or os.getenv("ORALSKOP_CHAT_MODEL", default_model)
        self.max_tokens = max_tokens
        if thinking is None:
            thinking = os.getenv("ORALSKOP_CHAT_THINKING", "").lower() in ("1", "true", "yes")
        self.thinking = thinking

    def reply(
        self,
        messages: list[dict],
        *,
        segmentation: dict | None = None,
        extra_context: str | None = None,
    ) -> dict:
        """Generate a reply. ``messages`` is the conversation as
        ``[{"role": "user"|"assistant", "content": str}, ...]``.
        """
        case_block = "Segmentation findings for this case:\n" + summarize_segmentation(segmentation)
        if extra_context:
            case_block += "\n\nAdditional context from the app:\n" + extra_context

        system = [
            # Stable instructions first (cacheable as the prompt grows); the
            # per-case findings go in a second, volatile block after it.
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": case_block},
        ]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": messages,
        }
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        resp = self.client.messages.create(**kwargs)

        reply_text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if resp.stop_reason == "refusal":
            reply_text = reply_text or (
                "I can't help with that request. For anything about your dental "
                "health, please consult a licensed dentist."
            )

        usage = resp.usage
        return {
            "reply": reply_text,
            "model": resp.model,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            },
        }
