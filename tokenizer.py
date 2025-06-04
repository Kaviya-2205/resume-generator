import json
import re

class Tokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def build_vocab(self, texts):
        for text in texts:
            for word in self._tokenize(text):
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def text_to_sequence(self, text, max_len=50):
        tokens = self._tokenize(text)
        sequence = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        if len(sequence) < max_len:
            sequence += [self.word2idx["<PAD>"]] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
        return sequence

    def save(self, path="tokenizer_vocab.json"):
        with open(path, "w") as f:
            json.dump(self.word2idx, f)

    def load(self, path="tokenizer_vocab.json"):
        with open(path, "r") as f:
            self.word2idx = json.load(f)
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)