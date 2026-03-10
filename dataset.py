import os
import torch
from typing import Union, List, Tuple
from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    def __init__(self):
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>'

        self.word2ind = {}
        self.ind2word = {}
        self.word_cnt = {}

        self.pad_ind = 3
        self.bos_ind = 1
        self.eos_ind = 2
        self.unk_ind = 0

    def build_vocab(self, sentences, min_freq = 2):
        self.word2ind = {
            self.pad_token : self.pad_ind,
            self.eos_token : self.eos_ind,
            self.bos_token : self.bos_ind,
            self.unk_token : self.unk_ind,
        }

        for sent in sentences:
            tokens = sent.split() # по условию уже токенизированы
            for token in tokens:
                self.word_cnt[token] = self.word_cnt.get(token, 0) + 1
            
        ind = len(self.word2ind)
        for word, cnt in self.word_cnt.items():
            if cnt >= min_freq:
                self.word2ind[word] = ind
                ind += 1

        self.ind2word = {ind : word for word, ind in self.word_cnt.items()}

    def encode(self, sentence):
        tokens = sentence.split()
        indicies = [self.bos_ind]
        for token in tokens:
            indicies.append(self.word2ind.get(token, self.unk_ind))

        indicies.append(self.eos_ind)
        return indicies
    
    def decode(self, indicies):
        return ' '.join([self.ind2word.get(ind, self.unk_token) for ind in indicies])
    
    def __len__(self):
        return len(self.word2ind)
    

class TranslationDataset(Dataset):
    def __init__(self, source_file, target_file, source_vocab = None, target_vocab = None):
        super().__init__()
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        with open(source_file) as f:
            self.source_sentences = [line.strip() for line in f]

        with open(target_file) as f:
            self.target_sentences = [line.strip() for line in f]

    def __len__(self):
        return len(self.source_sentences)
    
    def __getitem__(self, index):
        source_sentence = self.source_sentences[index]
        target_sentence = self.target_sentences[index]

        source_indices = self.source_vocab.encode(source_sentence)
        target_indices = self.target_vocab.encode(target_sentence)

        return torch.tensor(source_indices, dtype = torch.long), torch.tensor(target_indices, dtype = torch.long)
    
def padding_func(batch, pad_ind, device):
    source_batch, target_batch = zip(*batch)

    sources = []
    targets = []
    source_length = []
    target_length = []

    for source, target in zip(source_batch, target_batch):

        source_length.append(len(source))
        target_length.append(len(target))

        sources.append(torch.tensor(source, device=device))
        targets.append(torch.tensor(target, device=device))


    source_padded = pad_sequence(sources, padding_value=pad_ind, batch_first=True)
    target_padded = pad_sequence(targets, padding_value=pad_ind, batch_first=True)
    source_lengths = torch.tensor(source_length, device=device)
    target_lengths = torch.tensor(target_length, device=device)

    return source_padded, target_padded, source_lengths, target_lengths
