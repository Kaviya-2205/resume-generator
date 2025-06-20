# ResumeGenAIBackend/app/services/train.py

import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from app.utils.preprocess import TextPreprocessor
from app.models.gen_model import ResumeObjectiveGenerator

DATASET_PATH = "ResumeGenAIBackend/dataset.json"
MODEL_SAVE_PATH = "ResumeGenAIBackend/saved_models/model.pth"
BATCH_SIZE = 16
EPOCHS = 30 or 50
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
LEARNING_RATE = 0.003
MAX_LEN = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom PyTorch Dataset
class ResumeDataset(Dataset):
    def __init__(self, data, preprocessor):
        self.data = data
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        output_text = item["output"]["short_description"] + " " + item["output"]["objective"]
        input_ids = self.preprocessor.encode(output_text)
        return input_ids


def collate_fn(batch):
    batch = [b for b in batch if len(b) > 1]
    lengths = [len(x) for x in batch]
    padded = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return padded, lengths


def train_model():
    print("🚀 Loading dataset...")
    with open(DATASET_PATH, "r") as f:
        raw_data = json.load(f)

    all_texts = [
        sample["output"]["short_description"] + " " + sample["output"]["objective"]
        for sample in raw_data
    ]

    preprocessor = TextPreprocessor()
    preprocessor.build_vocab(all_texts)

    dataset = ResumeDataset(raw_data, preprocessor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = ResumeObjectiveGenerator(vocab_size=preprocessor.vocab_size(),
                                     embedding_dim=EMBEDDING_DIM,
                                     hidden_dim=HIDDEN_DIM).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("🧠 Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch, lengths in dataloader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"✅ Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab": preprocessor.token_to_id
    }, MODEL_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()