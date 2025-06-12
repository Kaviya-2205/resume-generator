# model.py
import torch
import torch.nn as nn

class ResumeModel(nn.Module):
    """
    A simulated LSTM-based language model for resume objective generation.
    """

    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2):
        super(ResumeModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        x: Tensor of token indices (batch_size, seq_len)
        Returns: Tensor of shape (batch_size, vocab_size)
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        final_hidden_state = lstm_out[:, -1, :]
        output = self.fc(final_hidden_state)
        return output