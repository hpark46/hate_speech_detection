import tensorflow as tf
import mitdeeplearning as mdl
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import html
from bs4 import BeautifulSoup
import regex as re
from Tokenizer import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def process_data():
    hate_data = pd.read_csv("/Users/hongjunpark/PycharmProjects/pythonProject5/hate_speech/hate_speech.csv",
                            names = ["Index", "Is_Hate", "Tweet"])
    hate_data.head()
    hate_data = np.array(hate_data)

    index = 0
    for sentence in hate_data[:,2]:                 # filtering hate_dataset
        hate_data[index,2] = filtering(sentence)
        index = index + 1

    train_set, validation_set, test_set = split_sets(hate_data)
    # print(test_set)
    tokenizer = Tokenizer()
    tokenizer.fit(train_set)
    tokenizer.fit_reverse(train_set)
    print(tokenizer.reverse_dict)





def split_sets(dataset):
    split_data = [0, 0, 0, 0, 0, 0, 0, 1, 1, 2]     # 0: train 1: validation 2: test

    split_list = [random.choice(split_data) for i in range(len(dataset))]

    # print(split_list.count(0)/len(split_list))
    # print(split_list.count(1)/len(split_list))
    # print(split_list.count(2)/len(split_list))

    train_set = []
    validation_set = []
    test_set = []

    index = 0
    for num in split_list:
        if num == 0:
            train_set.append(dataset[index])
        elif num == 1:
            validation_set.append(dataset[index])
        else:
            test_set.append(dataset[index])
        index = index + 1

    return train_set, validation_set, test_set







def filtering(input):
    holder = re.sub(r'#', ' #', input)
    holder = re.sub(r'@', ' @', holder)
    holder = re.sub(r'(http|https)\S+', '', holder)                                          # getting rid of url
    holder = BeautifulSoup(html.unescape(holder), 'html.parser').text                       # html parse
    holder = ''.join(filter(lambda x: x.isalnum() or x in [' ', '#', '@'], holder)).lower() # filtering
    return holder






if __name__ == '__main__':
    process_data()
    # print(filtering('~~Ruffled | Ntac Eileen Dahlia - Beautiful color combination of pink, orange, yellow &amp; white. A Coll http://t.co/H0dYEBvnZB'))