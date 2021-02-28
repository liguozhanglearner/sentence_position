import os
from io import open
import torch
import pandas as pd
from torch.autograd import Variable

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path,embeding_size):
        self.dictionary = Dictionary()
        self.embedding_size = embeding_size
        self.train, self.train_max_length= self.tokenize(os.path.join(path, 'squad_train.csv'),embeding_size)
        self.valid,self.valid_max_length = self.tokenize(os.path.join(path, 'squad_dev.csv'),embeding_size)
        # self.test,self.test_max_length = self.tokenize(os.path.join(path, 'squad_test.csv'),embeding_size)

    def char_tokenizer(sent):
        char_sent = ' '.join(list(sent))
        return char_sent

    def tokenize(self,path,embeding_size):
        try:
            for k in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:
                self.dictionary.add_word(k)
            max_length = 0
            df = pd.read_csv(path)
            for index, row in df.iterrows():
                sentence = row['risk_content'] + row['clause_content'] + row['risk_category']
                length = len(sentence)
                if length > max_length:
                    max_length = length
                for word in sentence:
                    self.dictionary.add_word(word)
            for i in range(1,max_length):
                self.dictionary.add_word(str(str(i)))
        except:
            print('read wrong')

        print('start construct tensor')

        sources,targets = [],[]
        for index,row in df.iterrows():
            source,target = [],[]
            temp_source = row['risk_content'] + row['clause_content'] + row['risk_category']
            temp_target = [str(row['start']),str(row['end'])]
            for word in temp_source:
                source.append(self.dictionary.word2idx[word])
            if len(source) < max_length:
                for j in range(0,max_length - len(source)):
                    source.append(self.dictionary.word2idx['<PAD>'])
            source.append(self.dictionary.word2idx['<EOS>'])

            for num in temp_target:
                target.append(self.dictionary.word2idx[num])
            # if len(target) < max_length:
            #     for j in range(0,max_length - len(target)):
            #         target.append(self.dictionary.word2idx['<PAD>'])
            # target.append(self.dictionary.word2idx['<EOS>'])

            sources.append(source)
            targets.append(target)

        assert len(sources) == len(targets)

        return (sources,targets),max_length