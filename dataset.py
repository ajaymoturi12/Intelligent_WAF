import os
from typing import List
from collections import Counter

import torch
from torch.utils.data import Dataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

from urllib import parse

class CSICDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocab=None, vocab_size=1000, min_frequency=1, special_tokens=["[UNK]","[CLS]","[PAD]"], tokenization_algorithm="bpe"):
        self.df = df

        # Export text content to csv for learning tokenization; apply BPE
        path = os.path.join('.', 'tokenization_input')
        
        self.df.to_csv(path_or_buf=path, columns=['content_for_tokenization'], index=False, header=False)

        if vocab == None:
            vocab = Vocab(vocab_size=vocab_size, min_frequency=min_frequency,
                           special_tokens=special_tokens,
                           tokenization_algorithm=tokenization_algorithm)
            vocab.build(corpus_files=[path])
        
        self.vocab = vocab
        
        self.encode_df()

    @staticmethod
    def process_df(df: pd.DataFrame):
        # Pre-process data by dropping rows without POST-Data or GET-Query
        get_mask, post_mask = df['GET-Query'].notna(), df['POST-Data'].notna()

        df.loc[get_mask,"content_for_tokenization"] = df.loc[get_mask,"GET-Query"]
        df.loc[post_mask,"content_for_tokenization"] = df.loc[post_mask,"POST-Data"]

        df = df[get_mask | post_mask]
        df = df.drop(columns=["GET-Query","POST-Data", "Accept-Charset", "Accept-Language", "Accept", "Cache-control", "Pragma", "Content-Type", "Host-Header", "Connection"])

        return df
    
    def encode_df(self):
        # Tokenize the GET-Query and POST-Data columns according to the subword vocabulary learned from BPE
        self.df["tokenized_ids"] = self.df["content_for_tokenization"].apply(lambda x: self.vocab.words2indices(x))
        self.df["tokenized"] = self.df["content_for_tokenization"].apply(lambda x: self.vocab.tokenize(x))
        self.df = self.df.drop(columns=["content_for_tokenization"])

        self.class_encoder, self.method_encoder = LabelEncoder(), LabelEncoder()
        self.df['Class'], self.df['Method'] = self.class_encoder.fit_transform(self.df['Class']), self.class_encoder.fit_transform(self.df['Method'])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        features = self.df.iloc[index].drop(['Class', 'User-Agent'])
        label = self.df.iloc[index]['Class']
        
        return features, label
        
class Vocab(object):
    def __init__(self, vocab_size=0, min_frequency=0, special_tokens: List[str]=[], unk_token="[UNK]", pad_token="[PAD]", tokenizer=None, tokenization_algorithm="bpe"):
        if tokenizer:
            self.tokenizer = tokenizer
            
            self.word2id = tokenizer.get_vocab()
            self.id2word = {v: k for k, v in self.word2id.items()}

            self.unk_id = self.word2id[unk_token]
        
        else:
            assert vocab_size > 0
            assert min_frequency > 0
            
            self.vocab_size = vocab_size
            self.min_frequency = min_frequency
            self.special_tokens = special_tokens
            self.unk_token = unk_token
            self.pad_token = pad_token
            self.tokenization_algorithm = tokenization_algorithm

    def build(self, corpus_files: List[str]):
        if self.tokenization_algorithm == 'bpe':
            tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
            trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=self.min_frequency, special_tokens=self.special_tokens)
            tokenizer.pre_tokenizer = ByteLevel()

            tokenizer.train(corpus_files, trainer)
            
            self.tokenizer = tokenizer
            self.word2id = tokenizer.get_vocab()
            self.id2word = {v: k for k, v in self.word2id.items()}
        
        elif self.tokenization_algorithm == 'vocab_map':
            self.word2id, self.id2word = dict(), dict()
            curr_id, self.min_frequency = 1,1
            counter = Counter()

            def add_to_vocab(token: str, ignore_cutoff=False):
                nonlocal curr_id
                if (ignore_cutoff or counter[token] >= self.min_frequency) and token not in self.word2id:
                    self.word2id[token] = curr_id
                    self.id2word[curr_id] = token

                    curr_id += 1
            
            for file_path in corpus_files:
                with open(file_path, 'r') as file:
                    for line in file:
                        tokens = Vocab.parse_req_body_or_params(line)
                        counter.update(tokens)
            
            unwanted_tokens = [' ','']
            for token in unwanted_tokens:
                if token in counter:
                    del counter[token]

            for token in self.special_tokens:
                add_to_vocab(token, ignore_cutoff=True)

            for token in set(counter.elements()):
                add_to_vocab(token)

        else:
            raise TypeError("Unsupported tokenization algorithm detected")

        self.unk_id = self.word2id[self.unk_token]
        self.pad_id = self.word2id[self.pad_token]

    def __getitem__(self, word):
        return self.word2id.get(word, self.unk_id)
    
    def __contains__(self, word):
        return word in self.word2id
    
    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self.word2id)

    def __repr__(self):
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        return self.id2word[wid]

    def save(self, file_path):
        self.tokenizer.save(path=file_path)

    def words2indices(self, content):
        if self.tokenization_algorithm == 'bpe':
            if type(content) == list:
                return [self.tokenizer.encode(row).ids for row in content]
            else:
                return self.tokenizer.encode(content).ids
            
        elif self.tokenization_algorithm == 'vocab_map':
            if type(content) == list:
                return [[self[token] for token in Vocab.parse_req_body_or_params(line)] for line in content]
            else:
                return [self[token] for token in Vocab.parse_req_body_or_params(content)]
        else:
            raise TypeError("Unsupported tokenization algorithm detected")
        
    def tokenize(self, content):
        if self.tokenization_algorithm == 'bpe':
            if type(content) == list:
                return [self.tokenizer.encode(row).tokens for row in content]
            else:
                return self.tokenizer.encode(content).tokens
            
        elif self.tokenization_algorithm == 'vocab_map':
            if type(content) == list:
                return [[token if self.__contains__(token) else self.unk_token for token in Vocab.parse_req_body_or_params(line)] for line in content]
            else:
                return [token if self.__contains__(token) else self.unk_token for token in Vocab.parse_req_body_or_params(content)]

        else:
            raise TypeError("Unsupported tokenization algorithm detected")
        
    @staticmethod
    def parse_req_body_or_params(line: str):
        parsed_line = parse.parse_qs(parse.unquote_plus(string=line))

        tokens = []
        for k, v in parsed_line.items():
            tokens.append(k)
            tokens.extend(v)

        return tokens

    @staticmethod
    def load(file_path: str):
        return Vocab(tokenizer=Tokenizer.from_file(file_path))