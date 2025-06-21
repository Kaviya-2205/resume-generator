# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.routes.generate import router as generate_router

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# Serve static assets
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

# API router
app.include_router(generate_router, prefix="/api")

# Serve HTML files
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/preview")
def preview():
    return FileResponse(FRONTEND_DIR / "preview.html")