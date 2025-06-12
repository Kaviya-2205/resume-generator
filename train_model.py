# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from model import ResumeModel
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("resume_data.csv")
texts = df["generated_resume"].tolist()

# Basic tokenizer and vocab builder
def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for text in texts:
        for word in text.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

vocab = build_vocab(texts)

# Save vocab
with open("tokenizer_vocab.json", "w") as f:
    json.dump(vocab, f)

# Tokenization function
def tokenize(text, vocab, max_len=50):
    tokens = [vocab.get(word, vocab["<UNK>"]) for word in text.lower().split()]
    tokens = tokens[:max_len] + [vocab["<PAD>"]] * (max_len - len(tokens))
    return tokens

# Prepare dataset
input_data = []
for text in texts:
    tokens = tokenize(text, vocab)
    input_data.append(tokens)
# targets (same as input for simulation)
X = torch.tensor(input_data)
y = torch.tensor(input_data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model setup
vocab_size = len(vocab)
model = ResumeModel(vocab_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (just for simulation)
epochs = 3
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train[:, 0])  #predicting first token
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "resume_model.pth")
print(" model training complete and model saved.")