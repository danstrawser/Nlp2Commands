__author__ = 'Dan'

from theano.compile.mode import FAST_COMPILE
from numpy import dtype
import numpy as np
import theano 
import theano.tensor as T
import theano.typed_list
from collections import OrderedDict
import pickle
import sys
import os
from theano import config

class DMN_No_Scan(object):

    # We take as input a string of "facts"
    def __init__(self, num_fact_hidden_units, number_word_classes, dimension_fact_embeddings, num_episode_hidden_units, max_number_of_facts_read):

        self.X_train, self.mask_train, self.question_train, self.question_train_mask, self.Y_train, self.X_test, self.mask_test, self.question_test, self.question_test_mask, self.Y_test, word2idx, idx2word, dimension_fact_embeddings, max_queslen = self.process_data("embeddings")
        number_word_classes = max(idx2word.keys(), key=int) + 1
        dimension_fact_embeddings = 7
        nv, de, cs = dimension_fact_embeddings, dimension_fact_embeddings, 1
        max_number_of_facts_read = 3
        nc = number_word_classes
        ne = number_word_classes # Using one hot, the number of embeddings is the same as the dimension of the fact embeddings
        nh = 7 # Dimension of the hidden layer
        num_hidden_units = nh
        num_hidden_units_facts = num_hidden_units
        num_hidden_units_episodes = num_hidden_units_facts
        num_hidden_units_questions = num_hidden_units_episodes











    def train(self):
        pass


    def process_data(self, type_of_embedding):

        filename_train = 'data/simple_dmn_theano/data_train_questioned.txt'
        filename_test = 'data/simple_dmn_theano/data_test_questioned.txt'

        X_train, mask_train, Question_train, Question_train_mask, Y_train, X_test, mask_test, Question_test, Question_test_mask, Y_test, max_queslen = [], [], [], [], [], [], [], [], [], [], 0

        cur_idx = 0
        word2idx = {}
        idx2word = {}

        max_mask_len = 0
        max_queslen = 0

        with open(filename_train, encoding='utf-8') as f:
            cur_sentence = []
            for line in f:
                if "?" not in line and "@" not in line:
                    cur_sentence.append(line.strip())
                else:
                    question_phrase = line[2:].split()
                    cur_question = []

                    if len(cur_sentence) > max_mask_len:
                        max_mask_len = len(cur_sentence)
                    if len(question_phrase) > max_queslen:
                        max_queslen = len(question_phrase)
                    for w in question_phrase:
                        cur_question.append(w)

                    Question_train.append((cur_question))
                    X_train.append(cur_sentence)
                    Y_train.append(next(f)[2:])
                    cur_sentence = []

        with open(filename_test, encoding='utf-8') as f:
            cur_sentence = []
            for line in f:
                if "?" not in line and "@" not in line:
                    cur_sentence.append(line.strip())
                else:
                    question_phrase = line[2:].split()
                    cur_question = []

                    if len(cur_sentence) > max_mask_len:
                        max_mask_len = len(cur_sentence)
                    if len(question_phrase) > max_queslen:
                        max_queslen = len(question_phrase)
                    for w in question_phrase:
                        cur_question.append(w)

                    Question_test.append((cur_question))
                    X_test.append(cur_sentence)
                    Y_test.append(next(f)[2:])
                    cur_sentence = []

        max_factlen = max_mask_len

        for l in X_train:
            cur_mask = np.zeros(max_mask_len, dtype='int32')
            cur_mask[0:len(l)] = 1
            mask_train.append(cur_mask)

        for l in X_test:
            cur_mask = np.zeros(max_mask_len, dtype="int32")
            cur_mask[0:len(l)] = 1
            mask_test.append(cur_mask)

        for l in Question_train:
            cur_mask = np.zeros(max_queslen, dtype='int32')
            cur_mask[0:len(l)] = 1
            Question_train_mask.append(cur_mask)

        for l in Question_test:
            cur_mask = np.zeros(max_queslen, dtype='int32')
            cur_mask[0:len(l)] = 1
            Question_test_mask.append(cur_mask)

        for l in X_train + X_test:
            for s in l:
                if s not in word2idx:
                    word2idx[s] = cur_idx
                    idx2word[cur_idx] = s
                    cur_idx += 1

        for y in Y_test + Y_train:
            if y not in word2idx:
                word2idx[y] = cur_idx
                idx2word[cur_idx] = y
                cur_idx += 1

        for q in Question_train + Question_test:
            for w in q:
                if w not in word2idx:
                    word2idx[w] = cur_idx
                    idx2word[cur_idx] = w
                    cur_idx += 1

        X_train_vec, Question_train_vec, Y_train_vec, X_test_vec, Question_test_vec, Y_test_vec = [], [], [], [], [], []

        for s in X_train:
            cur_sentence = []
            for f in s:
                if type_of_embedding == "embeddings":
                    new_vec = [word2idx[f]]
                else:
                    new_vec = np.zeros((len(word2idx)))
                    new_vec[word2idx[f]] = 1
                cur_sentence.append(new_vec)
            X_train_vec.append(np.asmatrix(cur_sentence))

        for s in X_test:
            cur_sentence = []
            for f in s:
                if type_of_embedding == "embeddings":
                    new_vec = [word2idx[f]]
                else:
                    new_vec = np.zeros((len(word2idx)))
                    new_vec[word2idx[f]] = 1
                cur_sentence.append(new_vec)
            X_test_vec.append(np.asmatrix(cur_sentence))

        for y in Y_train:
            if type_of_embedding == "embeddings":
                new_vec = word2idx[y]
            else:
                new_vec = np.zeros((len(word2idx)))
                new_vec[word2idx[y]] = 1
            Y_train_vec.append(new_vec)

        for y in Y_test:
            if type_of_embedding == "embeddings":
                new_vec = word2idx[y]
            else:
                new_vec = np.zeros((len(word2idx)))
                new_vec[word2idx[y]] = 1
            Y_test_vec.append(new_vec)

        for q in Question_train:
            cur_question = []
            for w in q:
                if type_of_embedding == "embeddings":
                    new_vec = [word2idx[w]]
                else:
                    assert(1 == 2) # Not implemented
                    #new_vec = np.zeros((len(word2idx)))
                    #new_vec[word2idx[f]] = 1
                cur_question.append(new_vec)
            Question_train_vec.append(np.asmatrix(cur_question))

        for q in Question_test:
            cur_question = []
            for w in q:
                if type_of_embedding == "embeddings":
                    new_vec = [word2idx[w]]
                else:
                    assert(1 == 2) # Not implemented
                    #new_vec = np.zeros((len(word2idx)))
                    #new_vec[word2idx[f]] = 1
                cur_question.append(new_vec)
            Question_test_vec.append(np.asmatrix(cur_question))

        # Ensure we have all the same length
        new_question_train_vec = []
        for q in Question_train_vec:
            for added_el in range(len(q), max_queslen):
                q = np.concatenate((q, [[0]]), axis=0)
            new_question_train_vec.append(q)
        Question_train_vec = new_question_train_vec

        new_question_test_vec = []
        for q in Question_test_vec:
            for added_el in range(len(q), max_queslen):
                q = np.concatenate((q, [[0]]), axis=0)
            new_question_test_vec.append(q)
        Question_test_vec = new_question_test_vec

        new_x_train_vec = []
        for f in X_train_vec:
            for added_el in range(len(f), max_factlen):
                f = np.concatenate((f, [[0]]), axis=0)
            new_x_train_vec.append(f)
        X_train_vec = new_x_train_vec

        new_x_test_vec = []
        for f in X_test_vec:
            for added_el in range(len(f), max_factlen):
                f = np.concatenate((f, [[0]]), axis=0)
            new_x_test_vec.append(f)
        X_test_vec = new_x_test_vec

        assert(len(X_test_vec) == len(Y_test_vec))

        return X_train_vec, mask_train, Question_train_vec, Question_train_mask, Y_train_vec, X_test_vec, mask_test, Question_test_vec, Question_test_mask, Y_test_vec, word2idx, idx2word, len(word2idx), max_queslen



