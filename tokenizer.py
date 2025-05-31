import re

class Tokenizer:
    def __init__(self):
        self.token_to_id = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.id_to_token = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.vocab_size = 4

    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def build_vocab(self, texts):
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.token_to_id:
                    self.token_to_id[token] = self.vocab_size
                    self.id_to_token[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]

    def decode(self, ids):
        return " ".join([self.id_to_token.get(i, "<UNK>") for i in ids])
