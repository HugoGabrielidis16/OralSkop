"""Run the serving app from inside a Jupyter notebook + expose it via ngrok.

A notebook kernel already owns an asyncio event loop, so uvicorn can't just
``.run()`` on the main thread. :func:`serve` starts uvicorn on a **daemon thread**
(the cell returns immediately, the kernel stays interactive) and — if ngrok is
configured — opens a public ``https`` URL you can hand to a friend.

Notebook usage (Python cell)::

    from oralskop.serve.notebook import serve
    server = serve(
        weights="runs/seg/deeplabv3_alphadent/best.pt",
        arch="deeplabv3_resnet50",      # only needed if the ckpt lacks metadata
        device="cuda",                  # or "cpu"
        ngrok_authtoken="<your token>", # from https://dashboard.ngrok.com (free)
    )
    # -> prints the public URL; share <url>/ (browser form) or POST to <url>/predict

    server.stop()                       # shut down the server + tunnel

Get a free ngrok authtoken at https://dashboard.ngrok.com/get-started/your-authtoken.
Without a token, the server still runs locally (http://localhost:PORT) — reachable
on your machine / LAN but not from the public internet.
"""

from __future__ import annotations

import threading
import time

import uvicorn

from oralskop.serve.app import create_app


class NotebookServer:
    """Handle for a uvicorn server running on a background thread (+ ngrok tunnel)."""

    def __init__(self, server: uvicorn.Server, thread: threading.Thread,
                 local_url: str, public_url: str | None, tunnel=None):
        self._server = server
        self._thread = thread
        self.local_url = local_url
        self.public_url = public_url
        self._tunnel = tunnel

    @property
    def url(self) -> str:
        """Best URL to share (public ngrok URL if available, else the local one)."""
        return self.public_url or self.local_url

    def stop(self) -> None:
        """Stop the server and close the ngrok tunnel (if any)."""
        if self._tunnel is not None:
            try:
                from pyngrok import ngrok
                ngrok.disconnect(self._tunnel.public_url)
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                print(f"(ngrok disconnect failed: {exc})")
        self._server.should_exit = True
        self._thread.join(timeout=5)
        print("Server stopped.")


def serve(
    weights,
    *,
    arch: str = "deeplabv3_resnet50",
    imgsz: int = 512,
    device: str = "cpu",
    conf: float = 0.5,
    min_area_frac: float = 0.0005,
    host: str = "0.0.0.0",
    port: int = 8000,
    ngrok_authtoken: str | None = None,
    use_ngrok: bool = True,
) -> NotebookServer:
    """Start the serving app on a background thread; return a :class:`NotebookServer`.

    Set ``ngrok_authtoken`` (or pre-configure ngrok) to get a public URL. Pass
    ``use_ngrok=False`` to stay local-only.
    """
    app = create_app(weights, arch=arch, imgsz=imgsz, device=device,
                     conf=conf, min_area_frac=min_area_frac)

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for the server to come up (or the thread to die) before returning.
    for _ in range(100):
        if server.started or not thread.is_alive():
            break
        time.sleep(0.1)
    if not thread.is_alive():
        raise RuntimeError(f"Server failed to start (is port {port} already in use?).")

    local_url = f"http://localhost:{port}"
    public_url, tunnel = None, None
    if use_ngrok:
        try:
            from pyngrok import ngrok
            if ngrok_authtoken:
                ngrok.set_auth_token(ngrok_authtoken)
            tunnel = ngrok.connect(port, "http")
            public_url = tunnel.public_url
        except Exception as exc:  # noqa: BLE001
            print(f"ngrok tunnel not opened ({exc}).\n"
                  f"Serving locally only at {local_url} . Pass a valid ngrok_authtoken "
                  f"for a public URL, or use_ngrok=False to silence this.")

    print(f"Local:  {local_url}")
    if public_url:
        print(f"Public: {public_url}        <- share this with your friend")
        print(f"        open {public_url}/ in a browser, or POST images to {public_url}/predict")
    print("Call server.stop() to shut down.")
    return NotebookServer(server, thread, local_url, public_url, tunnel)
