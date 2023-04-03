import logging
logger = logging.getLogger(__name__)

import os
from typing import Union
import numpy as np

class Dataset(object):
    def __init__(self, filepath: str = '') -> None:
        self.filepath = filepath        
        self.data = []
        self.train_idx = None
        self.validation_idx = None
        self.test_idx = None

    def load(self) -> bool:
        if self.filepath:
            with open(self.filepath, 'r') as f:
                lines = f.readlines()
            
            for i in range(0, len(lines), 3):
                if (len(lines[i].strip()) > 0) and (len(lines[i+1].strip()) > 0):
                    self.data.append((lines[i], lines[i+1]))                    
            
            return True
    
    def train_test_split(self, test_size: float, train_size: float, 
                            seed: Union[int, None] = None) -> None:
        total = len(self.data)
        self.train_idx = int(train_size * total)
        self.test_idx  = total  
        if (test_size + train_size) == 1.0:
            self.validation_idx = None                        
        else:
            test_count = int(test_size * total)
            validation_count = total - self.train_idx - test_count
            self.validation_idx = self.train_idx + validation_count
            
        if self.data:
            if seed:
                random_state = np.random.RandomState(seed)
            else:
                random_state = np.random.RandomState()
            random_state.shuffle(self.data)
            return
        else:
            logger.error('Please load data first before performing train test split.')
            return

    def save(self, directory: str) -> bool:
        with open(os.path.join(directory, 'data.dat'), 'w') as f:
            _ = [''.join(row) + '\n' for row in self.data]
            print(len(_))
            f.write(
                ''.join(_)
            )
        with open(os.path.join(directory, 'train.dat'), 'w') as f:
            _ = [''.join(row) + '\n' for row in self.data[:self.train_idx]]
            print(len(_))
            f.write(
                ''.join(_)
            ) 
        if self.validation_idx:
            with open(os.path.join(directory, 'validation.dat'), 'w') as f:
                _ = [''.join(row) + '\n' for row in self.data[self.train_idx:self.validation_idx]]
                print(len(_))
                f.write(
                    ''.join(_)
                ) 
            with open(os.path.join(directory, 'test.dat'), 'w') as f:
                _ = [''.join(row) + '\n' for row in self.data[self.validation_idx:self.test_idx]]
                print(len(_))
                f.write(
                    ''.join(_)
                )                 
        else:
            with open(os.path.join(directory, 'test.dat'), 'w') as f:
                _ = [''.join(row) + '\n' for row in self.data[self.train_idx:self.test_idx]]
                print(len(_))
                f.write(
                    ''.join(_)
                )             

