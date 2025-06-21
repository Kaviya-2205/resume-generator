# ResumeGenAIBackend/app/models/gen_model.py

import torch
import torch.nn as nn

class ResumeObjectiveGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2):
        super(ResumeObjectiveGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)  # (batch_size, seq_len, vocab_size)
        return logits, hidden