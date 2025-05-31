import torch
import torch.nn as nn
import torch.optim as optim
import json
import pickle
from tokenizer import Tokenizer
from model import ResumeModel
from torch.utils.data import DataLoader, Dataset

# Custom dataset
class ResumeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src = self.tokenizer.encode(item["resume"])
        tgt = self.tokenizer.encode(item["job_description"])
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

# Load data
with open("resume_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Tokenizer
tokenizer = Tokenizer()
tokenizer.build_vocab([item["resume"] + item["job_description"] for item in data])

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Model
model = ResumeModel(vocab_size=tokenizer.vocab_size, d_model=128, nhead=8, num_layers=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Dataset and DataLoader
dataset = ResumeDataset(data, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)

# Training loop
for epoch in range(1, 4):
    model.train()
    total_loss = 0

    for batch in dataloader:
        optimizer.zero_grad()
        src_batch, tgt_batch = zip(*batch)

        # Pad to same length
        max_src_len = max(len(x) for x in src_batch)
        max_tgt_len = max(len(x) for x in tgt_batch)

        src_batch = [torch.cat([x, torch.zeros(max_src_len - len(x), dtype=torch.long)]) for x in src_batch]
        tgt_batch = [torch.cat([x, torch.zeros(max_tgt_len - len(x), dtype=torch.long)]) for x in tgt_batch]

        src = torch.stack(src_batch)
        tgt = torch.stack(tgt_batch)

        outputs = model(src, tgt[:, :-1])
        loss = criterion(outputs.reshape(-1, tokenizer.vocab_size), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), f"resume_model_epoch_{epoch}.pth")
