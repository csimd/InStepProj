import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import np_utils
import tensorflow as tf
import pandas as pd
from gensim import models

model = ''
def parse_q_a(line, parse_corpus):
        curr = line.split('\t')
        q = curr[0]
        a = curr[1]
        score = curr[2].strip('\n')

        #If the question is not in the dictionary, make a new qa pair and store in the dict. Else just store a new answer-score tuple in the question's corresponding list
        if q not in parse_corpus:
                parse_corpus[q] = [[a, score]]
        else:
                parse_corpus[q].append([a, score])
        return parse_corpus


def make_3_layered_onion(parse_corpus):
	outer_layer = []
	for key in parse_corpus:
		inner_layer = [[ans[0].split()] + [ans[1]] for ans in parse_corpus[key]]
		outer_layer.append([key.split()] + inner_layer)

	return outer_layer

def tokenize_curr(parse_corpus):
    corpus_tokenized = []
    for key in parse_corpus:
        corpus_tokenized.append(key)
        for i in parse_corpus[key]:
            corpus_tokenized += i
    corpus_tokenized = [sentence.split() for sentence in corpus_tokenized]
    return corpus_tokenized

def update_model(model, corpus, parse_corpus, sz):
    for corp in corpus:
            parse_corpus = parse_q_a(corp, parse_corpus)

    corpus_tokenized = tokenize_curr(parse_corpus)
    windowSz = 7
    dictSize = len(corpus_tokenized)
    if model == '':
        model = models.Word2Vec(corpus_tokenized, size=100, window=windowSz, min_count=1, workers=4)
        model.save('ass-train')
        print(model)
        return model
    else:
        new_model = models.Word2Vec.load('ass-train')
        new_model.build_vocab(corpus_tokenized, update=True)
        new_model.train(corpus_tokenized, sz, epochs=10)
        new_model.save('ass-train')
        print(new_model)
        return new_model


firstQ = True
corpus = []
cur_Q = ''
line = ''
sz = 0
with open('SelQA-ass-train.txt', 'r') as f:
    while(f):
        line = f.readline() + '\n'
        if firstQ :
            cur_Q = line.split('?')[0]
            firstQ = False
            sz += len(line.split())
        elif line.split('?')[0] != cur_Q:
            model = update_model(model, corpus, {}, sz)
            cur_Q = line.split('?')[0]
            print(cur_Q)
            corpus = [line]
            sz = len(line.split())
        elif line == '': break
        else:
            corpus.append(line)
            sz += len(line.split())