# ResumeGenAIBackend/app/utils/dataset_generator.py

import random
import json
import os

names = ["Aarav", "Diya", "Kaviya", "Raj", "Anika", "Ishaan", "Meera", "Vivaan"]
domains = ["gmail.com", "yahoo.com", "outlook.com"]
skills_pool = ["Python", "JavaScript", "React", "Node.js", "PyTorch", "Machine Learning", "Data Analysis", "SQL"]
projects_pool = ["AI Resume Generator", "Stock Price Predictor", "Chatbot", "E-commerce Website", "Portfolio Builder"]
certificates_pool = ["Coursera AI", "AWS Certified", "Google Data Cert", "Meta Frontend Developer"]
job_titles = ["Software Engineer", "Machine Learning Engineer", "Data Analyst", "Full Stack Developer"]

short_desc_templates = [
    "A passionate and motivated learner aiming to grow in the field of {}.",
    "Dedicated to building impactful solutions in the {} industry.",
    "Self-driven individual excited about opportunities in {} and eager to contribute."
]

objective_templates = [
    "To secure a role as a {} where I can apply my skills in {} and contribute to innovative projects.",
    "Seeking a {} position that allows me to utilize my background in {} and grow as a professional.",
    "To obtain a challenging role in {} that leverages my strengths in {} and passion for technology."
]

job_description_tags = {
    "Machine Learning Engineer": ["PyTorch", "ML", "data", "models"],
    "Software Engineer": ["backend", "APIs", "databases", "scalable"],
    "Data Analyst": ["SQL", "reports", "dashboard", "insights"],
    "Full Stack Developer": ["frontend", "React", "Node.js", "UI"]
}


def generate_sample():
    name = random.choice(names)
    phone = "+91 9" + "".join([str(random.randint(0, 9)) for _ in range(9)])
    email = f"{name.lower()}@{random.choice(domains)}"
    
    skills = random.sample(skills_pool, k=3)
    projects = random.sample(projects_pool, k=2)
    certificates = random.sample(certificates_pool, k=2)
    
    job_title = random.choice(job_titles)
    job_description = f"We are looking for a {job_title} with experience in {', '.join(job_description_tags[job_title])}."

    short_description = random.choice(short_desc_templates).format(job_title.lower())
    objective = random.choice(objective_templates).format(job_title, ", ".join(skills))

    return {
        "input": {
            "name": name,
            "phone": phone,
            "email": email,
            "skills": skills,
            "projects": projects,
            "certificates": certificates,
            "short_resume_text": "Experienced with multiple tech stacks and real-world projects.",
            "job_description": job_description
        },
        "output": {
            "name": name,
            "phone": phone,
            "email": email,
            "short_description": short_description,
            "objective": objective
        }
    }


def generate_dataset(n_samples=5000 or 10000, save_path="ResumeGenAIBackend/dataset.json"):
    dataset = [generate_sample() for _ in range(n_samples)]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"✅ Generated dataset with {n_samples} samples at {save_path}")


if __name__ == "__main__":
    generate_dataset()