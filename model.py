import torch
import torch.nn as nn

class ResumeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super(ResumeModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)                              # [batch, seq_len, embed_dim]
        lstm_out, _ = self.lstm(embedded)                          # [batch, seq_len, hidden_dim]
        out = self.output(lstm_out)                                # [batch, seq_len, vocab_size]
        return out