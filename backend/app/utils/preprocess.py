# ResumeGenAIBackend/app/utils/preprocess.py

import re
from collections import Counter
import torch

class TextPreprocessor:
    def __init__(self, max_vocab_size=5000):
        self.max_vocab_size = max_vocab_size
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.id_to_token = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.vocab_built = False

    def tokenize(self, text):
        # Basic tokenization (lowercased words and punctuation separated)
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s.,!?]", "", text)
        tokens = text.strip().split()
        return tokens

    def build_vocab(self, texts):
        # Build vocabulary from list of text samples
        token_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counts.update(tokens)

        most_common = token_counts.most_common(self.max_vocab_size - len(self.token_to_id))
        for i, (token, _) in enumerate(most_common, start=len(self.token_to_id)):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        self.vocab_built = True

    def encode(self, text, add_special_tokens=True):
        tokens = self.tokenize(text)
        ids = [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]
        if add_special_tokens:
            ids = [self.token_to_id["<SOS>"]] + ids + [self.token_to_id["<EOS>"]]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        tokens = [self.id_to_token.get(idx, "<UNK>") for idx in ids]
        text = " ".join(tokens)
        return text.replace("<SOS>", "").replace("<EOS>", "").strip()

    def vocab_size(self):
        return len(self.token_to_id)