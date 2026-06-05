from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import auth, screenings, history

app = FastAPI(title="OralSkop API", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten before production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(screenings.router, prefix="/api", tags=["screenings"])
app.include_router(history.router, prefix="/api", tags=["history"])


@app.get("/health")
def health():
    return {"status": "ok"}
