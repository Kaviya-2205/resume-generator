import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader
from model import ResumeModel

# Tokenizer class
class Tokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inverse_vocab = {}

    def build_vocab(self, texts):
        idx = len(self.vocab)
        for text in texts:
            for word in text.lower().split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1
        self.inverse_vocab = {idx: word for word, idx in self.vocab.items()}

    def encode(self, text, max_len=50):
        tokens = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text.lower().split()]
        tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
        return tokens

    def save(self, path="tokenizer.pkl"):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.vocab, f)

# Dataset class
class ResumeDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=50):
        self.samples = []
        for item in data:
            # Use job_description as input and resume_text + objective as target
            input_text = item["job_description"]
            target_text = item["resume_text"] + " " + item["objective"]
            self.samples.append((tokenizer.encode(input_text, max_len), tokenizer.encode(target_text, max_len)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)

# Load data
with open("resume_data.json") as f:
    data = json.load(f)

# Build vocab from all texts combined (input + target)
all_texts = []
for item in data:
    all_texts.append(item["job_description"])
    all_texts.append(item["resume_text"])
    all_texts.append(item["objective"])

tokenizer = Tokenizer()
tokenizer.build_vocab(all_texts)
tokenizer.save()

# Dataset and DataLoader
dataset = ResumeDataset(data, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)  # batch size 2 for example

# Model initialization
vocab_size = len(tokenizer.vocab)
model = ResumeModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for inputs, targets in loader:
        outputs = model(inputs)  
        
        outputs = outputs.view(-1, vocab_size)
        targets = targets.view(-1)
        
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Save model
torch.save(model.state_dict(), "resume_model.pth")
print("✅ Model training complete. Saved as resume_model.pth")