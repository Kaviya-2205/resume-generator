# ResumeGenAIBackend/app/services/inference.py

import torch
from app.models.gen_model import ResumeObjectiveGenerator
from app.utils.preprocess import TextPreprocessor
import os

MODEL_PATH = "ResumeGenAIBackend/saved_models/gen_model.pth"
MAX_GEN_LEN = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and vocab
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    preprocessor = TextPreprocessor()
    preprocessor.token_to_id = checkpoint["vocab"]
    preprocessor.id_to_token = {v: k for k, v in preprocessor.token_to_id.items()}

    model = ResumeObjectiveGenerator(
        vocab_size=len(preprocessor.token_to_id)
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, preprocessor

# Generate text using greedy decoding
def generate_text(prompt_text, model, preprocessor):
    input_ids = preprocessor.encode(prompt_text, add_special_tokens=True).unsqueeze(0).to(device)
    generated_ids = [preprocessor.token_to_id["<SOS>"]]

    hidden = None
    for _ in range(MAX_GEN_LEN):
        input_tensor = torch.tensor([[generated_ids[-1]]], dtype=torch.long).to(device)
        output, hidden = model(input_tensor, hidden)
        next_token_id = output.argmax(2).item()

        if next_token_id == preprocessor.token_to_id["<EOS>"]:
            break
        generated_ids.append(next_token_id)

    return preprocessor.decode(generated_ids)

# Main public function
def generate_resume_output(user_input):
    """
    user_input = {
        "name": "...",
        "phone": "...",
        "email": "...",
        "skills": [...],
        "projects": [...],
        "certificates": [...],
        "short_resume_text": "...",
        "job_description": "..."
    }
    """
    model, preprocessor = load_model()

    prompt = (
    f"{user_input['short_resume_text']} "
    f"Skills: {', '.join(user_input['skills'])}. "
    f"Projects: {', '.join(user_input['projects'])}. "
    f"Certificates: {', '.join(user_input['certificates'])}. "
    f"Job: {user_input['job_description']}"
    )

    full_text = generate_text(prompt, model, preprocessor)

    # For now we return the same name/phone/email from input
    return {
        "name": user_input["name"],
        "phone": user_input["phone"],
        "email": user_input["email"],
        "short_description": full_text.split(".")[0].strip() + ".",
        "objective": ".".join(full_text.split(".")[1:]).strip()
    }