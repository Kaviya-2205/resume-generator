from flask import Flask, request, jsonify
import torch
from model import ResumeModel  # Your model class
from tokenizer import Tokenizer # Your tokenizer class

app = Flask(__name__)

# Load tokenizer and model
tokenizer = Tokenizer()
tokenizer.load(r"C:\copied desktop\New folder\ResumeCustomizerProject\tokenizer.py")  # Adjust path or load method as you have

vocab_size = len(tokenizer.token2id)
model = ResumeModel(vocab_size)
model.load_state_dict(torch.load(r"C:\copied desktop\New folder\ResumeCustomizerProject\model.py", map_location='cpu'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    # Tokenize input text (adjust based on your tokenizer)
    input_ids = tokenizer.encode(text)
    input_tensor = torch.tensor([input_ids])

    with torch.no_grad():
        output = model.generate(input_tensor)  # You may need to implement generate or use your model forward method

    # Decode model output to readable text (adjust as needed)
    result_text = tokenizer.decode(output[0].tolist())

    return jsonify({'result': result_text})

if __name__ == '__main__':
    app.run(debug=True)
    import pickle

# Save tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickel.load(f)