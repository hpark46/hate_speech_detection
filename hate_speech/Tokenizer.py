import numpy as np
import copy

class Tokenizer():
    def __init__(self):
        self.dictionary = {'d3f0u1t': 0}

    def pre_process(self, dataset):
        holder = []

        tweets = np.array(dataset, dtype=str)[:, 2]
        for sentence in tweets:
            holder.append(sentence.split())

        return holder


    def fit(self, dataset):
        holder = self.pre_process(dataset)
        flattened = []

        for listed in holder:
            for word in listed:
                flattened.append(word)

        enumerated = enumerate(sorted(set(flattened)), 1)
        self.dictionary.update({y: x for (x, y) in enumerated})

    def transform(self, dataset):
        tokenized = []
        for sentence in dataset:
            for words in sentence:



