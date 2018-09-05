import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

torch.manual_seed(1)

context_size = 4 # target 앞 뒤로 4개의 단어를 참조
embedding_dim = 300
num_epochs = 20

corpus_text = "This tutorial will walk you through the key ideas of deep learning programming using Pytorch." \
              " Many of the concepts (such as the computation graph abstraction and autograd) " \
              "are not unique to Pytorch and are relevant to any deep learning tool kit out there.".split()

class CBOW(nn.Module):
    def __init__(self, vocab_size, embeddig_size, context_size):
        super(CBOW, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(context_size*embedding_dim, 128)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        out = F.relu(self.fc1(self.embedding(x).view(1, -1)))
        out = self.fc2(out)
        out = F.log_softmax(out, dim=1)
        return out


