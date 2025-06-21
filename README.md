# Resume Generator AI 🧠📄

This project is an AI-powered resume objective customizer. It uses a *custom generative LSTM model* built from scratch in *PyTorch*, without any pre-trained logic, to create tailored resume objectives based on job descriptions and candidate input.

## ✨ Features

- Custom-built *generative AI model* (not classification-based)
- AI-generated resume objectives using:
  - User's personal info, skills, projects, certifications, etc.
  - Job description
  - Uploaded or pasted resume
- Clean dark-themed frontend with multiple resume templates
- Resume download as PDF or Word

## 📦 Installation

> (Recommended) Create and activate a virtual environment first
pip install -r requirements.txt

If you get a email-validator error while using FastAPI/Pydantic, install with:
pip install "pydantic[email]"

## 🚀 Run Backend Server
uvicorn app.main:app --reload
Access the API at:
- Swagger UI: http://127.0.0.1:8000/docs
- Endpoint: POST /api/generate

## 🧠 Model Training (Optional)
To retrain the model:
python -m app.services.train
To test output from trained model:
python test_output_sample.py

## 📫 Contact
Maintained by Kaviya  
Built with custom PyTorch + FastAPI GenAI pipeline.