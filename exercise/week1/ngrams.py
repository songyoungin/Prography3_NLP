import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

    lr = 0.01
    num_epochs = 100

    criterion = nn.NLLLoss()
    model = NGram(vocab_size, embeddig_dim, context_size)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        loss_val = 0.
        for context, target in trigrams:
            context_idx = [word2idx[w] for w in context]
            context_idx = torch.LongTensor(context_idx)
            target_idx = torch.LongTensor([word2idx[target]])

            optimizer.zero_grad()
            outputs = model(context_idx)
            loss = criterion(outputs, target_idx)
            loss.backward()
            optimizer.step()

            loss_val += loss.item()
        print("[%d/%d] loss: %.3f" % (epoch, num_epochs, loss_val / len(trigrams)))

    print("Learning finished!")
    model.cpu()
    torch.save(model, "weights/ngrams.pth")

def test(word2idx):

    idx2word = {v: k for k, v in word2idx.items()}

    model = torch.load("weights/ngrams.pth")
    print(model)

    test_input = "make my old".split()
    test_input_idx = [word2idx[token] for token in test_input[:-1]]
    test_input_idx = torch.LongTensor(test_input_idx)
    test_target_idx = [word2idx[test_input[-1]]]

    outputs = model(test_input_idx)
    outputs, idx = torch.max(outputs, dim=1)

    print("predicted:", idx2word[idx.data[0].item()], " actual:", idx2word[test_target_idx[0]])

def main():

    test_sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()

    trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
                for i in range(len(test_sentence) - 2)]

    word2idx = {}
    for token in test_sentence:
        if token not in word2idx:
            word2idx[token] = len(word2idx)

    train(trigrams, word2idx)
    test(word2idx)

if __name__ == "__main__":
    main()
