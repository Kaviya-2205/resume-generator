# ResumeGenAIBackend/app/routes/generate.py

from fastapi import APIRouter
from app.schemas import ResumeInput, ResumeOutput
from app.services.inference import generate_resume_output

router = APIRouter()

@router.post("/generate", response_model=ResumeOutput)
def generate_resume_objective(input_data: ResumeInput):
    output = generate_resume_output(input_data.dict())
    return output