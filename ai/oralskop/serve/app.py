"""FastAPI app exposing the trained torchseg model as an HTTP endpoint.

Endpoints
---------
* ``GET  /``               — a tiny HTML upload form (drag a photo, see the overlay).
* ``GET  /health``         — liveness probe.
* ``GET  /info``           — loaded checkpoint metadata + class names.
* ``POST /predict``        — multipart ``file=@photo.jpg`` -> JSON detections.
* ``POST /predict/overlay``— multipart ``file=@photo.jpg`` -> annotated PNG.

Build the app with :func:`create_app` (used by the CLI and by the notebook helper)::

    from oralskop.serve.app import create_app
    app = create_app("runs/seg/deeplabv3_alphadent/best.pt", arch="deeplabv3_resnet50")

Run standalone (outside a notebook)::

    uv run python -m oralskop.serve.app \
        --weights runs/seg/deeplabv3_alphadent/best.pt --arch deeplabv3_resnet50 --port 8000
"""

from __future__ import annotations

import argparse
import io
import logging
import traceback

from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

_log = logging.getLogger("oralskop.serve")

from oralskop.serve.model import SegModel


# ----------------------------------------------------------------- chat schemas
class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """One stateless chat turn: the full conversation so far plus the case's
    stored segmentation result (the JSON returned by ``POST /predict``)."""

    messages: list[ChatTurn] = Field(..., min_length=1)
    segmentation: dict | None = None
    extra_context: str | None = None


class ChatReply(BaseModel):
    reply: str
    model: str
    usage: dict


def _get_chat_service(app: FastAPI):
    """Lazily build (and cache) the Claude-backed chat service on the app.

    Deferred so the segmentation server still starts without an API key / the
    anthropic package — only ``/chat`` needs them.
    """
    svc = getattr(app.state, "chat_service", None)
    if svc is None:
        from oralskop.serve.chat import ChatService

        svc = ChatService()  # raises ChatUnavailable if dep/key missing
        app.state.chat_service = svc
    return svc

_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>OralSkop</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}
 h1{font-size:1.4rem} .meta{color:#666;font-size:.85rem}
 #out img{max-width:100%;border-radius:8px;margin-top:1rem}
 button{padding:.5rem 1rem;font-size:1rem;cursor:pointer}
 pre{background:#f5f5f5;padding:1rem;border-radius:8px;overflow:auto}
</style></head><body>
<h1>OralSkop — dental segmentation</h1>
<p class="meta">Upload an intraoral / phone photo. Returns the model's segmentation overlay.</p>
<input type="file" id="f" accept="image/*">
<button onclick="go()">Analyze</button>
<div id="out"></div>
<script>
async function go(){
  const f=document.getElementById('f').files[0];
  if(!f){alert('Pick an image first');return;}
  const out=document.getElementById('out'); out.innerHTML='Running…';
  const fd=new FormData(); fd.append('file',f);
  const [img,js]=await Promise.all([
    fetch('/predict/overlay',{method:'POST',body:fd}).then(r=>r.blob()),
    fetch('/predict',{method:'POST',body:fd}).then(r=>r.json())
  ]);
  out.innerHTML='<img src="'+URL.createObjectURL(img)+'">'+
    '<pre>'+JSON.stringify(js,null,2)+'</pre>';
}
</script></body></html>"""


def create_app(weights, *, arch="deeplabv3_resnet50", imgsz=512, device="cpu",
               conf=0.5, min_area_frac=0.0005) -> FastAPI:
    """Build a FastAPI app serving the torchseg checkpoint at `weights`."""
    model = SegModel(weights, arch=arch, imgsz=imgsz, device=device,
                     conf=conf, min_area_frac=min_area_frac)
    app = FastAPI(title="OralSkop dental segmentation", version="0.1.0")

    async def _read(file: UploadFile) -> bytes:
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(415, f"Expected an image, got {file.content_type!r}.")
        data = await file.read()
        if not data:
            raise HTTPException(400, "Empty upload.")
        return data

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _INDEX_HTML

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/info")
    async def info():
        return model.info()

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        data = await _read(file)
        try:
            return JSONResponse(model.predict(data))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:  # surface the real cause instead of an opaque 500
            _log.error("predict failed:\n%s", traceback.format_exc())
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")

    @app.post("/predict/overlay")
    async def predict_overlay(file: UploadFile = File(...)):
        data = await _read(file)
        try:
            png = model.predict_overlay(data)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except Exception as exc:
            _log.error("predict/overlay failed:\n%s", traceback.format_exc())
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")
        return StreamingResponse(io.BytesIO(png), media_type="image/png")

    @app.post("/chat", response_model=ChatReply)
    def chat(req: ChatRequest):  # sync def -> Starlette runs it in a threadpool
        from oralskop.serve.chat import ChatUnavailable

        if req.messages[-1].role != "user":
            raise HTTPException(400, "The last message must have role 'user'.")
        try:
            svc = _get_chat_service(app)
        except ChatUnavailable as exc:
            raise HTTPException(503, str(exc))
        try:
            return svc.reply(
                [m.model_dump() for m in req.messages],
                segmentation=req.segmentation,
                extra_context=req.extra_context,
            )
        except Exception as exc:  # surface the real cause like the other routes
            _log.error("chat failed:\n%s", traceback.format_exc())
            raise HTTPException(500, f"{type(exc).__name__}: {exc}")

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve a torchseg checkpoint over HTTP.")
    p.add_argument("--weights", required=True, help="Checkpoint (best.pt/last.pt).")
    p.add_argument("--arch", default="deeplabv3_resnet50",
                   help="Fallback arch if the checkpoint lacks metadata.")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--device", default="cpu")
    p.add_argument("--conf", type=float, default=0.5, help="Min mean-softmax conf per detection.")
    p.add_argument("--min-area-frac", type=float, default=0.0005,
                   help="Drop components smaller than this fraction of the image.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ngrok", action="store_true",
                   help="Open a public ngrok tunnel to this server (needs the `serve` extra).")
    p.add_argument("--ngrok-token", default=None,
                   help="ngrok authtoken (else uses the NGROK_AUTHTOKEN env var / saved config).")
    return p.parse_args(argv)


def _open_ngrok(port: int, token: str | None) -> None:
    """Open an ngrok tunnel to `port` and print the public URL to stdout."""
    from pyngrok import ngrok

    if token:
        ngrok.set_auth_token(token)
    ngrok.kill()  # drop any stale tunnel (free tier allows only one at a time)
    url = ngrok.connect(port, "http").public_url
    print(f"Public URL: {url}", flush=True)
    print(f"Open form: {url}/", flush=True)


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = parse_args(argv)
    app = create_app(args.weights, arch=args.arch, imgsz=args.imgsz, device=args.device,
                     conf=args.conf, min_area_frac=args.min_area_frac)
    if args.ngrok:
        _open_ngrok(args.port, args.ngrok_token)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
