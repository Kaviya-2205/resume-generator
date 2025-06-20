# ResumeGenAIBackend/app/schemas.py

from pydantic import BaseModel, EmailStr
from typing import List

class ResumeInput(BaseModel):
    name: str
    phone: str
    email: EmailStr
    skills: List[str]
    projects: List[str]
    certificates: List[str]
    job_description: str
    short_resume_text: str

class ResumeOutput(BaseModel):
    name: str
    phone: str
    email: EmailStr
    short_description: str
    objective: str