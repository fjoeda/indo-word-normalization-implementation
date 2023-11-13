import torch
import string


class CharacterEncoder:
    def __init__(self):
        alpha = string.ascii_lowercase + string.digits + string.punctuation
        self.char2idx = {
            "<unk>": 0,
            "<pad>": 1,
            "<sos>": 2,
            "<eos>": 3,
        }
        self.idx2char = ["<unk>", "<pad>", "<sos>", "<eos>"]
        for c in alpha:
            if c not in self.idx2char:
                self.char2idx[c] = len(self.idx2char)
                self.idx2char.append(c)

    def get_total_char(self):
        return len(self.idx2char)

    def encode(self, word):
        word = str(word)
        word = word.lower().strip()
        vector = [self.char2idx['<sos>']]
        for c in word:
            vector.append(self.char2idx.get(c, 1))

        vector.append(self.char2idx['<eos>'])
        
        return torch.tensor(vector, dtype=torch.long)