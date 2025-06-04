import torch
import torch.nn as nn
from flask import Flask, request, render_template
from model import ResumeModel
import pickle
import json

# Load tokenizer vocab
with open("tokenizer_vocab.json") as f:
    vocab = json.load(f)
inverse_vocab = {int(v): k for k, v in vocab.items()}
vocab_size = len(vocab)

# Load model
model = ResumeModel(vocab_size)
model.load_state_dict(torch.load("resume_model.pth"))
model.eval()

# Tokenizer class for encoding input
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text, max_len=50):
        tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.lower().split()]
        tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
        return tokens

    def decode(self, token_ids):
        words = [inverse_vocab.get(int(idx), "<UNK>") for idx in token_ids if idx != 0]
        return ' '.join(words)

tokenizer = Tokenizer(vocab)

# Inference
def generate_resume(job_description, max_len=50):
    input_ids = tokenizer.encode(job_description, max_len)
    input_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_ids = torch.argmax(outputs, dim=2).squeeze().tolist()
    return tokenizer.decode(predicted_ids)

# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    generated_resume = ""
    if request.method == "POST":
        job_description = request.form["job_description"]
        generated_resume = generate_resume(job_description)
    return render_template("index.html", result=generated_resume)

if __name__ == "__main__":
    app.run(debug=True)