import torch
from collections import Counter

class BPEncoder:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.vocab = {}

    def fit(self, texts):
        # Count character frequencies
        char_counts = Counter("".join(texts))
        
        # Initialize the vocabulary with individual characters
        self.vocab = {char: freq for char, freq in char_counts.items()}

        for _ in range(self.num_merges):
            pairs = self.get_pairs()
            if not pairs:
                break

            # Find the most frequent pair
            most_frequent_pair = max(pairs, key=lambda p: self.vocab.get(p, 0))
            if most_frequent_pair not in self.vocab:
                break

            # Merge the most frequent pair
            self.vocab["".join(most_frequent_pair)] = self.vocab.pop(most_frequent_pair)

    def get_pairs(self):
        pairs = set()
        for word, freq in self.vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs.add((symbols[i], symbols[i + 1]))
        return pairs

    def encode(self, text):
        symbols = list(text)
        encoded = []
        while symbols:
            for i in range(len(symbols), 0, -1):
                subword = symbols[:i]
                if subword in self.vocab:
                    encoded.append(self.vocab[subword])
                    symbols = symbols[i:]
                    break
            else:
                # If no subword is found in the vocabulary, treat it as an unknown token
                encoded.append(self.vocab["<unk>"])
                symbols = symbols[1:]
        return torch.tensor(encoded)

    def decode(self, tensor):
        decoded = ""
        for token in tensor.tolist():
            decoded += list(self.vocab.keys())[list(self.vocab.values()).index(token)]
        return decoded


