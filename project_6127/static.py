import logging
import os

LOGGER_PRINT_LEVEL = logging.DEBUG
DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRECTORY_DATA = os.path.join(DIRECTORY, 'data')
DIRECTORY_OUTPUT = os.path.join(DIRECTORY, 'output')

class Config(object):
    seed = 1111
    cudnn_benchmark = True
    cuda_is_available = False

    # vectorizer configs
    max_words = None # determined from dataset if None
    min_freq = 5 # no filter if None 
    start_end_tokens = True
    max_len = None # determined from dataset if None

    # embedding configs
    emsize = 512
    embedding_size = 512

    # model configs
    cell = "GRU"
    learning_rate = 0.001
    num_of_exams = 3
    
    nlayers = 1    
    epochs = 10
    batch_size = 64
    dropout = 0
    bidirectional = True
    max_grad_norm = 10   

    relative_data_path = '/data/train.dat'
    relative_dev_path = '/data/dev.dat'
    relative_gen_path = '/data/fake%d.dat'
     
    

class ConfigTest(Config):    
    emsize = 3
    lr = 1
    epochs = 3
    batch_size = 2
    relative_data_path = '/data/haha.dat'
    relative_dev_path = '/data/haha.dat'
    relative_gen_path = '/data/fake%d.dat'
    max_grad_norm = 1
    min_freq = 0
