import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

word2idx = {}
char2idx = {}


def get_index_of_max(input):
    index = 0
    for i in range(1, len(input)):
        if input[i] > input[index]:
            index = i
    return index


def get_max_prob_result(input, ix_to_tag):
    return ix_to_tag[get_index_of_max(input)]


def get_char_seq(word, char2idx):
    idxs = []
    for car in word:
        idxs.append(char2idx[car])
    return idxs


def get_seq(seq, word2idx, char2idx):
    idxs = []
    for w in seq:
        idxs.append((word2idx[w], get_char_seq(w, char2idx)))
    return idxs


def get_target(seq, tag2idx):
    idxs = []
    for w in seq:
        idxs.append(tag2idx[w])
    return torch.LongTensor(idxs).to(device)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

for sent, tags in training_data:
    for word in sent:
        if word not in word2idx:
            word2idx[word] = len(word2idx)
        for car in word:
            if car not in char2idx:
                char2idx[car] = len(char2idx)

tag2idx = {"DET": 0, "NN": 1, "V": 2}
idx2tag = {v:k for k, v in tag2idx.items()}

char_embedding_dim = 3
char_hidden_dim = 3

word_embedding_dim = 6
hidden_dim = 6

char_vocab_size = len(char2idx)
word_vocab_size = len(word2idx)
tagset_size = len(tag2idx)

class LSTMTagger(nn.Module):

    def __init__(self, word_embedding_dim, char_embedding_dim,
                 hidden_dim, char_hidden_dim,
                 word_vocab_size, char_vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.char_embedding_dim = char_embedding_dim

        self.char_emb = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim)

        self.word_emb = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.word_lstm = nn.LSTM(word_embedding_dim + char_hidden_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.hidden = self.init_hidden(hidden_dim)
        self.char_hidden = self.init_hidden(char_hidden_dim)

    def init_hidden(self, dim):
        return (torch.zeros(1, 1, dim).to(device),
                torch.zeros(1, 1, dim).to(device))

    def forward(self, sentence):
        word_idxs = []
        char_lstm_result = []
        for word in sentence:
            words = word[0]
            chars = word[1]

            self.char_hidden = self.init_hidden(char_embedding_dim)
            word_idxs.append(words)
            char_idx = torch.LongTensor(chars).to(device)

            char_emb_out = self.char_emb(char_idx)
            char_emb_out = char_emb_out.view(len(chars), 1, char_embedding_dim)

            char_lstm_out, self.char_hidden = self.char_lstm(char_emb_out, self.char_hidden)
            char_lstm_result.append(char_lstm_out[-1])

        char_lstm_result = torch.stack(char_lstm_result)

        word_idxs = torch.LongTensor(word_idxs).to(device)
        word_emb_out = self.word_emb(word_idxs)
        word_emb_out = word_emb_out.view(len(sentence), 1, word_embedding_dim)

        lstm_in = torch.cat((word_emb_out, char_lstm_result), 2)

        lstm_out, self.hidden = self.word_lstm(lstm_in, self.hidden)
        lstm_out = lstm_out.view(len(sentence), -1)
        tag_out = self.hidden2tag(lstm_out)
        tag_scores = F.log_softmax(tag_out)
        return tag_scores

model = LSTMTagger(word_embedding_dim, char_embedding_dim,
                   hidden_dim, char_hidden_dim,
                   len(word2idx), len(char2idx), len(tag2idx))
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()

        model.hidden = model.init_hidden(hidden_dim)

        sentence_in = get_seq(sentence, word2idx, char2idx)

        targets = get_target(tags, tag2idx)

        tag_scores = model(sentence_in)

        loss = criterion(tag_scores, targets)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 30 == 0:
        print("[%d/%d] loss:%.3f" % (epoch+1, 300, loss.item()))

torch.save(model.state_dict(), "speech_tagging.pth")
# ======================= TEST

test_sentence = training_data[0][0]
inputs = get_seq(test_sentence, word2idx, char2idx)
tag_scores = model(inputs)
for i in range(len(test_sentence)):
    print('{}: {}'.format(test_sentence[i], get_max_prob_result(tag_scores[i].data.numpy(), idx2tag)))