# test_output_sample.py

from app.services.inference import generate_resume_output

sample_input = {
    "name": "Kaviya R",
    "phone": "+91 9876543210",
    "email": "kaviya@example.com",
    "skills": ["Python", "Deep Learning", "NLP"],
    "projects": ["Resume Parser using NLP", "AI Job Matcher"],
    "certificates": ["Coursera Deep Learning", "Udemy NLP Specialization"],
    "short_resume_text": "Motivated graduate with hands-on experience in AI projects and Python development.",
    "job_description": "Looking for a Machine Learning Engineer with experience in NLP and model deployment."
}

output = generate_resume_output(sample_input)

print("\n📤 AI-Generated Resume Output:")
print("Name:", output["name"])
print("Phone:", output["phone"])
print("Email:", output["email"])
print("Short Description:\n", output["short_description"])
print("Objective:\n", output["objective"])