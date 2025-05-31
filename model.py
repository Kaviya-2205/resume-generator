import torch
import torch.nn as nn

class ResumeModel(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super(ResumeModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt):
        # Encoder
        embedded_src = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded_src)

        # Decoder
        embedded_tgt = self.embedding(tgt)
        output, _ = self.decoder(embedded_tgt, (hidden, cell))
        logits = self.fc(output)
        return logits
