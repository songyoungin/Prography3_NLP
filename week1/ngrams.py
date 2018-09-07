import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle
import os
from tensorboardX import SummaryWriter

torch.manual_seed(1)

class NGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGram, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * context_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        emb = self.embedding(x).view(1, -1)
        # print(emb.shape)
        out = F.relu(self.linear1(emb))
        # print(out.shape)
        out = self.linear2(out)
        # print(out.shape)
        prob = F.log_softmax(out, dim=1)
        # print(prob.shape)
        return prob

def train(trigrams, word2idx):
    context_size = 2
    embeddig_dim = 10
    vocab_size = len(word2idx)

    lr = 0.05
    num_epochs = 10

    criterion = nn.NLLLoss()
    model = NGram(vocab_size, embeddig_dim, context_size)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print("Learning started!!!")
    for epoch in range(num_epochs):
        for step, (context, target) in enumerate(trigrams):
            context_idx = [word2idx[w] for w in context]
            context_idx = torch.LongTensor(context_idx)
            target_idx = torch.LongTensor([word2idx[target]])

            optimizer.zero_grad()
            outputs = model(context_idx)
            loss = criterion(outputs, target_idx)
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                print("[%d/%d] [%d/%d] loss: %.3f" % (epoch+1, num_epochs, step+1, len(trigrams), loss.item()))

    print("Learning finished!")
    model.cpu()

    os.makedirs("weights", exist_ok=True)
    torch.save(model, "weights/ngrams_newsCorpus.pth")

def test(trigrams, word2idx):
    idx2word = {v: k for k, v in word2idx.items()}
    model = torch.load("weights/ngrams_newsCorpus.pth")
    print(model)

    test_input = trigrams[2]
    test_input_idx = [word2idx[token] for token in test_input[0]]
    test_input_idx = torch.LongTensor(test_input_idx)
    test_target_idx = [word2idx[test_input[1][0]]]

    model.eval()
    outputs = model(test_input_idx)
    outputs, idx = torch.max(outputs, dim=1)

    print("predicted:", idx2word[idx.data[0].item()], " actual:", idx2word[test_target_idx[0]])

def visualize(word2idx):
    logdir = "tensorboard/ngrams"
    os.makedirs(logdir, exist_ok=True)

    model = torch.load("weights/ngrams_newsCorpus.pth")
    idx2word = {v: k for k, v in word2idx.items()}

    mat = model.embedding.weight.data
    groups = [idx2word[i] for i in range(len(word2idx))]

    print(mat.shape)
    print(len(groups))

    writer = SummaryWriter(logdir)
    writer.add_embedding(mat, metadata=groups)
    writer.close()

def main():
    # load data from pickle file
    with open("pickles/news_tokenize.pkl", "rb") as f:
        tokenized = pickle.load(f)

    trigrams = []
    for sentence in tokenized:
        for i in range(len(sentence)-2):
            trigrams.append(([sentence[i], sentence[i+1]], sentence[i+2]))

    word2idx = {}
    for sentence in tokenized:
        for token in sentence:
            if token not in word2idx:
                word2idx[token] = len(word2idx)

    train(trigrams, word2idx)
    test(trigrams, word2idx)
    visualize(word2idx)

if __name__ == "__main__":
    main()