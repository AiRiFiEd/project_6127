import logging
from ..static import LOGGER_PRINT_LEVEL
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(LOGGER_PRINT_LEVEL)

import os
from typing import Union, List
from collections import Counter
import numpy as np

import torch
from torch.utils.data import Dataset

class Data(object):
    def __init__(self, filepath: str = '') -> None:
        self.filepath = filepath        
        self.data = [] # [[title, abstract], [title, abstract]]
        self.train_idx = None
        self.validation_idx = None
        self.test_idx = None

    def load(self) -> bool:
        logger.debug('loading dataset...')
        if self.filepath:
            logger.debug('filepath: {filepath}'.format(filepath=self.filepath))
            with open(self.filepath, 'r') as f:
                lines = f.readlines()            
            logger.info('# of lines in raw data: {num_of_lines}'.format(num_of_lines=len(lines)))

            counter = 0     
            for i in range(0, len(lines), 3):
                if (len(lines[i].strip()) > 0) and (len(lines[i+1].strip()) > 0):
                    self.data.append((lines[i], lines[i+1]))                    
                else:
                    counter += 1
            logger.info('# of lines in loaded data: {num_of_lines}'.format(num_of_lines=len(self.data)))
            logger.info('# of lines excluded: {num_of_lines}'.format(num_of_lines=counter))
            logger.debug('loading dataset completed.')
            return True
    
    def train_test_split(self, test_size: float, train_size: float, 
                            seed: Union[int, None] = None) -> None:
        logger.debug('performing train test split...')
        total = len(self.data)
        
        self.train_idx = int(train_size * total)
        self.test_idx  = total  
        if (test_size + train_size) == 1.0:
            self.validation_idx = None                        
        else:
            test_count = int(test_size * total)
            validation_count = total - self.train_idx - test_count
            self.validation_idx = self.train_idx + validation_count
        
        logger.info('total # of rows: {num_of_rows}'.format(num_of_rows=total))
        logger.info('# of training data: {num_of_rows}'.format(num_of_rows=self.train_idx))
        logger.info('# of validation data: {num_of_rows}'.format(num_of_rows=(self.validation_idx-self.train_idx)))
        logger.info('# of test data: {num_of_rows}'.format(num_of_rows=(self.test_idx-self.validation_idx)))                    
        
        if self.data:
            if seed:
                random_state = np.random.RandomState(seed)
            else:
                random_state = np.random.RandomState()
            random_state.shuffle(self.data)
            logger.debug('performing train test split completed.')
            return
        else:
            logger.error('Please load data first before performing train test split.')
            return

    def save(self, directory: str) -> bool:
        with open(os.path.join(directory, 'data.dat'), 'w') as f:
            _ = [''.join(row) + '\n' for row in self.data]            
            f.write(
                ''.join(_)
            )
        with open(os.path.join(directory, 'train.dat'), 'w') as f:
            _ = [''.join(row) + '\n' for row in self.data[:self.train_idx]]            
            f.write(
                ''.join(_)
            ) 
        if self.validation_idx:
            with open(os.path.join(directory, 'validation.dat'), 'w') as f:
                _ = [''.join(row) + '\n' for row in self.data[self.train_idx:self.validation_idx]]                
                f.write(
                    ''.join(_)
                ) 
            with open(os.path.join(directory, 'test.dat'), 'w') as f:
                _ = [''.join(row) + '\n' for row in self.data[self.validation_idx:self.test_idx]]                
                f.write(
                    ''.join(_)
                )                 
        else:
            with open(os.path.join(directory, 'test.dat'), 'w') as f:
                _ = [''.join(row) + '\n' for row in self.data[self.train_idx:self.test_idx]]                
                f.write(
                    ''.join(_)
                )             

class Vectorizer(object):
    def __init__(self, max_words: Union[int, None] = None,
                    min_frequency: Union[int, None] = None,
                    start_end_tokens: bool = True,
                    max_len: Union[int, None] = None):
        
        self.max_words = max_words
        self.min_frequency = min_frequency
        self.start_end_tokens = start_end_tokens
        self.max_len = max_len

        self.vocabulary = dict()
        self.vocabulary_excluded = dict()
        self.vocabulary_size = 0 
        self.word2idx = dict()
        self.idx2word = dict()

    def _find_max_sentence_length(self, corpus: List, template: bool) -> None:
        logger.debug('finding max length from corpus...')
        if not template:
            self.max_len = max(len(sent) 
                                for document in corpus
                                for sent in document)
        else:
            self.max_len = max(len(sent)
                                for sent in corpus)
            
        logger.info('max length without start and end tokens: {max_len}'.format(max_len=self.max_len))        
        if self.start_end_tokens:
            self.max_len += 2
            logger.info('max length with start and end tokens: {max_len}'.format(max_len=self.max_len))
        
        logger.debug('finding max length from corpus completed.')
        
    def _build_vocabulary(self, corpus: List, template: bool) -> None:
        """
        Instantiates `Vectorizer.vocabulary` with a dictionary
        of {word:freq}.
        """
        logger.debug('building vocabulary...')
        if not template:
            vocabulary = Counter(word for document in corpus 
                                 for sent in document
                                 for word in sent)
            
        else:
            vocabulary = Counter(word for sent in corpus
                                    for word in sent)        
        
        logger.info('# of words in vocabulary: {num_of_words}'.format(num_of_words=len(vocabulary)))

        if self.max_words:
            vocabulary = {
                word:freq for word, freq in vocabulary.most_common(self.max_words)
            }
            logger.info('# of words in vocabulary (capped): {num_of_words}'.format(num_of_words=len(vocabulary)))
        
        if self.min_frequency:
            for word, freq in vocabulary.items():
                if freq >= self.min_frequency:
                    self.vocabulary[word] = freq
                else:
                    self.vocabulary_excluded[word] = freq

        logger.info('# of words meeting min freq requirement: {num_of_words}'.format(num_of_words=len(self.vocabulary)))
        logger.info('# of words excluded: {num_of_words}'.format(num_of_words=len(self.vocabulary_excluded)))

        self.vocabulary_size = len(self.vocabulary) + 2 # why?
        if self.start_end_tokens:
            self.vocabulary_size += 2
       
        logger.info('# of words in vocabulary (final): {num_of_words}'.format(num_of_words=len(self.vocabulary)))
        logger.info('vocabulary_size: {vocabulary_size}'.format(vocabulary_size=self.vocabulary_size))

        logger.debug('building vocabulary completed.')

    def _build_word_index(self) -> None:
        """
        Instantiates `Vectorizer.word2idx` and `Vectorizer.idx2word`
        with each word in `Vectorizer.vocabulary` and its associated
        index.
        """
        logger.debug('building word index...')
        self.word2idx['<UNK>'] = 3
        self.word2idx['<PAD>'] = 0

        if self.start_end_tokens:
            self.word2idx['<EOS>'] = 1
            self.word2idx['<SOS>'] = 2
        
        offset = len(self.word2idx)
        for idx, word in enumerate(self.vocabulary):
            self.word2idx[word] = idx + offset
        
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}
        logger.debug('building word index completed.')

    def transform_sentence(self, sentence: str) -> List:
        vector = [self.word2idx.get(word, self.word2idx['<UNK>']) 
                    for word in sentence]
        return vector

    def fit(self, corpus: List, template: bool = False) -> None:
        logger.debug('fitting corpus...')
        if not self.max_len:
            self._find_max_sentence_length(corpus, template)            
        self._build_vocabulary(corpus, template)
        self._build_word_index()        
        logger.debug('fitting corpus completed.')

    def transform(self, corpus: List, template: bool = False) -> List:
        """
        Converts corpus to list of indices.
        """
        logger.debug('creating word index vector...')
        vcorpus = []
        if not template:
            for document in corpus:
                vcorpus.append([self.transform_sentence(sentence) for sentence in document])
        else:
            vcorpus.extend([self.transform_sentence(sentence) for sentence in corpus])
        
        logger.debug('creating word index vector completed.')
        return vcorpus

class Abstracts(Dataset):
    def __init__(self, max_len: int = 200, 
                 use_cuda: bool = torch.cuda.is_available()) -> None:
        self.max_len = max_len # allowed max length of abstracts 
        self.use_cuda = use_cuda

        self.head_len = 0
        self.abs_len = 0 # actual max length of abstracts, capped at `max_len`
        
        self.data = None
        #self.corpus = self._read_corpus(path)
        #self.data = self._vectorize_corpus()
        #self._initalcorpus()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        len_s, len_t, source, target = self.data[index]
        source = torch.LongTensor(source)
        target = torch.LongTensor(target)

        if self.use_cuda:
            source = source.cuda()
            target = target.cuda()
        
        return source, target, len_s

    def pad_sentence_vector(self, vector: List, max_len: int) -> List:
        padding = max_len - len(vector)
        vector.extend([0] * padding)
        return vector
    
    def _tokenize_word(self, sentence: str) -> List:  
        """
        Simple tokenization just by splitting by space.
        """      
        return [word for word in sentence.split() if word]        

    def _read_corpus(self, data: Data) -> List:
        """
        Converts data from [[title, abstract], [title, abstract], ...]
        to [[
                [title_word_1, title_word_2, ...], [abstract_word_1, abstract_word_2, ...]
            ],
            [
                [title_word_1, title_word_2, ...], [abstract_word_1, abstract_word_2, ...]
            ],
            ...
        ]
        """
        logger.debug('reading corpus...')
        logger.info('# of rows in dataset: {num_of_rows}'.format(num_of_rows=len(data.data)))
        corpus = []        
        counter = 0
        for pair in data.data:            
            title = self._tokenize_word(pair[0])
            abstract = self._tokenize_word(pair[1])                
            if title and abstract:                
                corpus.append([title, abstract])
            else:
                counter += 1

        logger.info('# of rows in corpus: {num_of_rows}'.format(num_of_rows=len(corpus)))
        logger.info('# of rows excluded: {num_of_rows}'.format(num_of_rows=counter))
        
        logger.debug('reading corpus completed.')
        return corpus
    
    def _vectorize_corpus(self, corpus: List, vectorizer: Vectorizer):
        """
        Converts corpus to list of indices.
        """
        logger.debug('vectorizing corpus...')
        if not vectorizer.vocabulary:            
            vectorizer.fit(corpus)            
        transformed = vectorizer.transform(corpus)
        logger.debug('vectorizing corpus completed.')
        return transformed
    
    def _initalcorpus(self, vectorized_corpus: List,
                        vectorizer: Vectorizer):
        """
        Apply padding to titles and abstracts with zeros to 
        make them of equal length.
        """
        # determine max length of titles and abstracts
        # pad titles and abstracts
        old = []
        for i in vectorized_corpus:
            source = i[0] # list of indices for title
            target = i[1] # list of indices for abstract
            if len(source) > self.head_len:
                self.head_len = len(source)

            if len(target) <= self.max_len:
                if len(target) > self.abs_len:
                    self.abs_len = len(target)
            else:
                target = target[:self.max_len-1]
                target.append(vectorizer.word2idx['<EOS>'])
                self.abs_len = len(target)
            old.append((source[1:-1], target))
        old.sort(key=lambda x: len(x[0]), reverse=True) # descending
        
        corpus = []
        for source, target in old:
            team = [
                        len(source), 
                        len(target), 
                        self.pad_sentence_vector(source, self.head_len), 
                        self.pad_sentence_vector(target, self.abs_len)
                    ]
            corpus.append(team)        
        self.data = corpus

    def process_corpus(self, data: Data, vectorizer: Vectorizer) -> None:
        corpus = self._read_corpus(data)
        corpus = self._vectorize_corpus(corpus, vectorizer)
        self._initalcorpus(corpus, vectorizer)

