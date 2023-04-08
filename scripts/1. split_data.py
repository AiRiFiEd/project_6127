import logging 
logger = logging.getLogger(__name__)

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import project_6127.data.preprocessing as preprocessing
import project_6127.static as static

if __name__ == '__main__':
    directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    directory_data = os.path.join(
        directory,
        'data')
    filepath = os.path.join(directory_data, 'acl_titles_and_abstracts.txt')
    logger.debug(filepath)    

    data = preprocessing.Data(filepath)
    data.load()
    config = static.Config()
    data.train_test_split(0.1, 0.8, config.seed)
    data.save(directory_data)