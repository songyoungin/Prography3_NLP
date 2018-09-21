
# coding: utf-8

# In[8]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[10]:


SOS_token = 0 # Start Of String
EOS_token = 1 # End of String
UNK_token = 2 # Unknown word

class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2idx = {'SOS': 0, 'EOS': 1, 'UNK': 2}
        self.word2count = {}
        self.idx2word = {0: 'SOS', 1: 'EOS', 2: 'UNK'}
        self.n_words = 3
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
        
    def add_word(self, word):
        # word2idx는 SOS, EOS, UNK token 없이 단어만 포함 --> idx 3부터 시작(0, 1, 2은 token 자리로 남겨둠)
        # idx2word는 SOS, EOS, UNK token 포함한 채로 시작 --> idx 0부터 시작
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.word2count[word] = 1
            self.idx2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    # Language 객체 내용 확인 위한 테스트 함수
    def get_infos(self):
        print('name:', self.name)
        print('word2idx:', self.word2idx)
        print('word2count:', self.word2count)
        print('idx2word:', self.idx2word)
        print('n_words:', self.n_words)


# In[11]:


# unicode 문자열 --> ASCII 코드로 변환
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 소문자 변환, trim, 알파벳 아닌 문자 제외
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip()) # lowercase
    s = re.sub(r"([.!?])", r" \1", s)     # trim
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# In[12]:


def read_langs(lang1, lang2, reverse=False):
    print("reading lines...")
    
    # txt 파일 읽어와서 line 단위로 잘라내기
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').        read().strip().split('\n')
        
    # print(lines)
    
    # line 단위를 lang 별로 끊어낸 다음에 normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    
    # print(pairs)
    
    # reverse pairs
    # 논문에서 소개된 input sentence의 단어의 순서를 거꾸로 넣는 것과는 별개, 단순히 input-output의 순서쌍을 반대로 하는 것
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    
    return input_lang, output_lang, pairs


# In[13]:


MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filter_pair(p):
    return len(p[0].split()) < MAX_LENGTH and len(p[1].split()) < MAX_LENGTH and p[0].startswith(eng_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


# In[14]:


def prepare_data(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse)
    print("read %d sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("trimmed to %d sentence pairs" % len(pairs))
    print("counting words...")
    
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
        
    print("counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    print("\nInput language object information:")
    input_lang.get_infos()
    
    print("\nTarget language object information:")
    output_lang.get_infos()
    
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepare_data(lang1='eng', lang2='fra', reverse=False)
print(random.choice(pairs))


# In[15]:


# Encoder model
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

# Decoder model
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x).view(1, 1, -1)
        output = F.relu(embedded)
        output, hidden = self.gru(output, hidden)
        
        output = self.fc(output[0])
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    
# Attention decoder model
class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_prob=0.1, max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.max_length = max_length
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size*2, self.max_length)
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.embedding(x).view(1, 1, -1)
        embedded = self.dropout(embedded)
        
        attn_input = torch.cat((embedded[0], hidden[0]), dim=1)
        
        attn_weight = F.softmax(self.attn(attn_input), dim=1)
        attn_applied = torch.bmm(attn_weight.unsqueeze(0), encoder_outputs.unsqueeze(0))
        
        attn_combine_input = torch.cat((embedded[0], attn_applied[0]), 1)
    
        output = self.attn_combine(attn_combine_input).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        
        output = self.fc(output[0])
        output = F.log_softmax(output, dim=1)
        
        return output, hidden, attn_weight

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[16]:


def idxs_from_sentence(lang, sentence):
    out = []
    for word in sentence.split():
        if word in lang.word2idx:
            out.append(lang.word2idx[word])
        else:
            out.append(lang.word2idx['UNK'])
    return out

def tensor_from_sentence(lang, sentence):
    idxs = idxs_from_sentence(lang, sentence)
    idxs.append(EOS_token)
    return torch.tensor(idxs, dtype=torch.long, device=device).view(-1, 1)

def tensors_from_pair(pair):
    input_tensor = tensor_from_sentence(input_lang, pair[0])
    target_tensor = tensor_from_sentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


# In[17]:


teacher_forcing_ratio = 0.5

# unit forward-backward function
# return unit loss
def unit_trainer(input_tensor, target_tensor, 
         encoder, decoder,
         encoder_optimizer, decoder_optimizer,
         criterion, max_length=MAX_LENGTH):
    
    encoder_hidden = encoder.init_hidden()
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)
    
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    
    loss = 0.
    
    '''
    input: length, 1
    target: length, 1

    ** ENCODER **
    embedding: 1, 1, D
    GRU: 1, 1, D

    final encoder ouptut: MAX_LENGTH, D

    ** DECODER **
    embedding: 1, 1, D
    attention input(concat embedded[0], hidden[0]:
    1, 2*D
    attention weight: 1, MAX_LENGTH
    attention applied: 1, 1, D
    attention combine input(concat embedded[0], attention applied[0]) : 1, 2*D
    attention combine output: 1, 1, D
    GRU: 1, 1, D
    FC: 1, OUTPUT_LANG.n_words
    '''
    
    # make encoder output list
    for idx in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[idx], encoder_hidden)
        encoder_outputs[idx] = encoder_output[0, 0]

    
    decoder_input = torch.tensor([[SOS_token]], device=device)    
    decoder_hidden = encoder_hidden
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for idx in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[idx])
            decoder_input = target_tensor[idx]         
    else:
        for idx in range(target_length):
            decoder_output, decoder_hidden, decodr_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            top_value, top_idx = decoder_output.topk(1)
            decoder_input = top_idx.squeeze().detach()
            loss += criterion(decoder_output, target_tensor[idx])
            
            if decoder_input.item() == EOS_token:
                break
             
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item() / target_length


# In[18]:


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# In[19]:


import numpy as np

def trainer(encoder, decoder, num_epochs,
           log_interval, lr, reverse_sequence=False):
    
    start_time = time.time()
    
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=lr)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=lr)
    
    training_pairs = [tensors_from_pair(random.choice(pairs)) for i in range(num_epochs)]
    criterion = nn.NLLLoss()
    
    print("Learning started!!!")
    
    for epoch in range(num_epochs):
        losses = []
        training_pair = training_pairs[epoch]
        input_tensor = training_pair[0]
        
        # 단어들의 순서를 반대로 입력 --> 논문에서 나온 reverse input
        if reverse_sequence:
            reverse_arr = np.flip(input_tensor.detach().cpu().numpy(), 0).copy()
            input_tensor = torch.from_numpy(reverse_arr)
            
        target_tensor = training_pair[1]
        
        loss = unit_trainer(input_tensor, target_tensor,
                           encoder, decoder,
                           encoder_optimizer, decoder_optimizer, criterion)
        losses.append(loss)
        
        if (epoch+1) % log_interval == 0:
            avg_loss = np.mean(losses)
            print("%s ([%d/%d] %.3f%%) %.3f"
                 % (timeSince(start_time, (epoch+1) / num_epochs),
                   epoch+1, num_epochs,  float(epoch+1) / num_epochs * 100, avg_loss))
            
    print("Learning finished!!!")
    
    torch.save(encoder, "eng-fra/encoder.pth")
    torch.save(decoder, "eng-fra/decoder.pth")


# In[21]:


hidden_size = 256

encoder = Encoder(input_lang.n_words, hidden_size).to(device)
attn_decoder = AttnDecoder(hidden_size, output_lang.n_words, dropout_prob=0.1).to(device)

# encoder.load_state_dict(torch.load('eng-fra/encoder.pth', map_location=lambda storage, loc: storage))
# attn_decoder.load_state_dict(torch.load('eng-fra/decoder.pth', map_location=lambda storage, loc: storage))

# trainer(encoder, attn_decoder, num_epochs=50000, log_interval=1000, lr=0.01, reverse_sequence=False)


# In[ ]:


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    # set model test mode
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # seq2seq 모델의 한계: 
        # training vocabulary 안에 존재하는 word에 대해서만 forwarding이 가능.
        # 처음 보는 word에 대해서는 <UKN> token 처리함으로서 정보 손실.
        # print(sentence)

        input_tensor = tensor_from_sentence(input_lang, sentence)
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.init_hidden()
        
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        
        # forward to encoder network
        for idx in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[idx], encoder_hidden)
            encoder_outputs[idx] = encoder_output[0, 0]
            
        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        
        for idx in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[idx] = decoder_attention.data
            top_value, top_idx = decoder_output.data.topk(1)
            
            if top_idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.idx2word[top_idx.item()])
            
            decoder_input = top_idx.squeeze().detach()
            
        return decoded_words, decoder_attentions[:idx+1]


# In[ ]:


def random_evaluate(encoder, decoder, samples=10):
    for idx in range(samples):
        pair = random.choice(pairs)
        print("input:", pair[0])
        print("target:", pair[1])
        
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        
        print("output:", output_sentence)
        print()


# In[ ]:

encoder = torch.load('eng-fra/encoder.pth', map_location=lambda storage, loc: storage)
attn_decoder = torch.load('eng-fra/decoder.pth', map_location=lambda storage, loc: storage)

random_evaluate(encoder, attn_decoder)

decoded_word, _ = evaluate(encoder, attn_decoder, 'my name is youngin song')
print(decoded_word)
print(' '.join(decoded_word))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(encoder, attn_decoder, input_sentence):
    output_words, attentions = evaluate(
        encoder, attn_decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention(encoder, attn_decoder, "i m thirty years older than you .")

evaluateAndShowAttention(encoder, attn_decoder, "they re waiting for you .")