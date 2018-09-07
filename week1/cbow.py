import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import pickle

torch.manual_seed(1)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * context_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        emb = self.embedding(x).view(1, -1)
        out = F.relu(self.linear1(emb))
        out = self.linear2(out)
        prob = F.log_softmax(out, dim=1)
        return prob

def train(cbows, word2idx):
    context_size = 2
    embeddig_dim = 10
    vocab_size = len(word2idx)

    lr = 0.05
    num_epochs = 10

    criterion = nn.NLLLoss()
    model = CBOW(vocab_size, embeddig_dim, context_size*2)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print("Learning started!!!")
    for epoch in range(num_epochs):
        for step, (context, target) in enumerate(cbows):
            context_idx = [word2idx[w] for w in context]
            context_idx = torch.LongTensor(context_idx)
            target_idx = torch.LongTensor([word2idx[target[0]]])

            optimizer.zero_grad()
            outputs = model(context_idx)
            loss = criterion(outputs, target_idx)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print("[%d/%d] [%d/%d] loss: %.3f" % (epoch+1, num_epochs, step+1, len(cbows), loss.item()))

    print("Learning finished!")
    model.cpu()
    torch.save(model, "weights/cbow_newsCorpus.pth")


def test(word2idx):
    idx2word = {v: k for k, v in word2idx.items()}

    model = torch.load("weights/ngrams_newsCorpus.pth")
    print(model)

    test_input = "make my old".split()
    test_input_idx = [word2idx[token] for token in test_input[:-1]]
    test_input_idx = torch.LongTensor(test_input_idx)
    test_target_idx = [word2idx[test_input[-1]]]

    outputs = model(test_input_idx)
    outputs, idx = torch.max(outputs, dim=1)

    print("predicted:", idx2word[idx.data[0].item()], " actual:", idx2word[test_target_idx[0]])


def main():
    # load data from csv file
    data_path = "abcnews-date-text.csv"
    data = pd.read_csv(data_path, engine='python')
    data = data.values[:500, 1]
    tokenized = [sentence.split() for sentence in data]

    context_size = 2

    cbows = []
    for sentence in tokenized:
        for i in range(context_size, len(sentence)-context_size):
            context = [sentence[i-2], sentence[i-1], sentence[i+1], sentence[i+2]]
            target = [sentence[i]]

            cbows.append((context, target))

    print(len(cbows))
    word2idx = {}
    for sentence in tokenized:
        for token in sentence:
            if token not in word2idx:
                word2idx[token] = len(word2idx)

    with open("pickles/cbows.pkl", "wb") as f:
        pickle.dump(cbows, f)

    with open("pickles/word2idx_cbow.pkl", "wb") as f:
        pickle.dump(word2idx, f)

    print("Saving pickles complete!!!")

    with open("pickles/cbows.pkl", "rb") as f:
        cbows = pickle.load(f)

    with open("pickles/word2idx_cbow.pkl", "rb") as f:
        word2idx = pickle.load(f)

    train(cbows, word2idx)

if __name__ == "__main__":
    main()
    test()
