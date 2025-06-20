# ResumeGenAIBackend/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.generate import router as generate_router

app = FastAPI(
    title="Resume Generator AI",
    description="Generates personalized resume objectives and summaries using a trained LSTM model.",
    version="1.0"
)

# Allow CORS (adjust origins as needed for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register the API route
app.include_router(generate_router, prefix="/api")