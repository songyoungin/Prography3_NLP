import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pickle

from collections import Counter
flatten = lambda l: [item for sublist in l for item in sublist]

# u: center word
# v: context(target) word
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()

        self.u_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.v_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        self.embedding_dim = embedding_dim

        self.init_embeddings()

    def init_embeddings(self):
        range = 0.5 / self.embedding_dim
        self.u_embeddings.weight.data.uniform_(-range, range)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        losses = []
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        pos_score = torch.mul(emb_u, emb_v).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)

        losses.append(sum(pos_score))

        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.sum(neg_score, dim=1)
        neg_score = F.logsigmoid(-1 * neg_score)

        losses.append(sum(neg_score))

        return -1 * sum(losses)

def get_one_hot(word_idx, vocab_size):
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x.long()

def train():
    pass

def main():
    data = ["I love you",
            "I am a girl",
            "I am reading a book",
            "My dad is a teacher",
            "You are a boy",
            "My hobby is playing piano"]

    data = [sentence.split() for sentence in data]

    # make vocabulary
    word2idx = {}
    for sentence in data:
        for w in sentence:
            if w not in word2idx:
                word2idx[w] = len(word2idx)
    idx2word = {v: k for k, v in word2idx.items()}

    # make word pairs
    window_size = 2
    idx_pairs = []
    for sentence in data:
        word_idxs = [word2idx[w] for w in sentence]
        for center_word_pos in range(len(word_idxs)):
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w

                if context_word_pos < 0 or context_word_pos >= len(word_idxs) or center_word_pos == context_word_pos:
                    continue

                idx_pairs.append((word_idxs[center_word_pos], word_idxs[context_word_pos]))


if __name__ == "__main__":
    main()


