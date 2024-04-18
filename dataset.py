import os
from typing import List

import torch
from torch.utils.data import Dataset

import pandas as pd

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

class CSICDataset(Dataset):
    def __init__(self, csv_path: str, vocab_size: int, min_frequency: int, special_tokens=["[UNK]","[CLS]"]):
        self.df = pd.read_csv(csv_path)
        self.process_df()

        # Export text content to csv for learning tokenization; apply BPE
        path = os.path.dirname(csv_path)
        path = os.path.join(path, 'tokenization_input')
        
        self.df.to_csv(path_or_buf=path, columns=['content_for_tokenization'], index=False, header=False)

        vocab = Vocab(vocab_size=vocab_size, min_frequency=min_frequency,
                           special_tokens=special_tokens)

        vocab.build(corpus_files=[path])
        self.vocab = vocab
        
        self.encode_df()

    def process_df(self):
        # Pre-process data by dropping rows without POST-Data or GET-Query
        get_mask, post_mask = self.df['GET-Query'].notna(), self.df['POST-Data'].notna()

        self.df.loc[get_mask,"content_for_tokenization"] = self.df.loc[get_mask,"GET-Query"]
        self.df.loc[post_mask,"content_for_tokenization"] = self.df.loc[post_mask,"POST-Data"]

        self.df = self.df[get_mask | post_mask]
        self.df = self.df.drop(columns=["GET-Query","POST-Data"])
    
    def encode_df(self):
        self.df["tokenized_ids"] = self.df["content_for_tokenization"].apply(lambda x: self.vocab.words2indices(x))
        self.df["tokenized"] = self.df["content_for_tokenization"].apply(lambda x: self.vocab.tokenize(x))

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        features = self.df.iloc[index].drop('Class')
        label = self.df.iloc[index]['Class']
        
        return features, label
        
class Vocab(object):
    def __init__(self, vocab_size=0, min_frequency=0, special_tokens: List[str]=[], unk_token="[UNK]", tokenizer=None):
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

    def build(self, corpus_files: List[str], unk_token="[UNK]"):
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(vocab_size=self.vocab_size, min_frequency=self.min_frequency, special_tokens=self.special_tokens)
        tokenizer.pre_tokenizer = ByteLevel()

        tokenizer.train(corpus_files, trainer)
        
        self.tokenizer = tokenizer
        self.word2id = tokenizer.get_vocab()
        self.id2word = {v: k for k, v in self.word2id.items()}

        self.unk_id = self.word2id[unk_token]

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
        if type(content) == list:
            return [self.tokenizer.encode(row).ids for row in content]
        else:
            return self.tokenizer.encode(content).ids
        
    def tokenize(self, content):
        if type(content) == list:
            return [self.tokenizer.encode(row).tokens for row in content]
        else:
            return self.tokenizer.encode(content).tokens

    @staticmethod
    def load(file_path: str):
        return Vocab(tokenizer=Tokenizer.from_file(file_path))