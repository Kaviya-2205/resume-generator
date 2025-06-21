# ResumeGenAIBackend/app/routes/generate.py

from fastapi import APIRouter
from app.schemas import ResumeInput, ResumeOutput
from app.services.inference import generate_resume_output
import logging

# ✅ Fix 1: Use correct variable _name_ (double underscores)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate", response_model=ResumeOutput)
def generate_resume_objective(input_data: ResumeInput):
    logger.info("Received resume input: %s", input_data.dict())
    output = generate_resume_output(input_data.dict())
    return output