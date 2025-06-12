import re

class Tokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def fit_on_texts(self, texts):
        for text in texts:
            for word in re.findall(r"\b\w+\b", text.lower()):
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1

    def texts_to_sequences(self, texts, max_len=100):
        sequences = []
        for text in texts:
            tokens = [self.word2idx.get(w, 1) for w in re.findall(r"\b\w+\b", text.lower())]
            if len(tokens) < max_len:
                tokens += [0] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
            sequences.append(tokens)
        return sequences

    def sequences_to_text(self, sequence):
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in sequence if idx != 0])