import numpy as np
import os
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk
from copy import deepcopy
import re
from scipy import sparse as sp
import random

class BabiProcessor(object):

    def __init__(self, type):
        self.type = type
        self.LEN_BABI_ARTICLE = 15
        self.filename_train = 'data/babi_tasks/tasks_1-20_v1-2/en/qa1_single-supporting-fact_train.txt'
        self.filename_test = 'data/babi_tasks/tasks_1-20_v1-2/en/qa1_single-supporting-fact_test.txt'

    def process(self):
        if self.type == "babi_simple":
            X_train_words, Y_train_words, max_seqlen = self._gen_babi_simple_vecs(self.filename_train, max_seqlen=0)
            X_test_words, Y_test_words, max_seqlen = self._gen_babi_simple_vecs(self.filename_test, max_seqlen)
            X_train, Y_train, mask_train, X_test, Y_test, mask_test, word2idx, idx2word = self._gen_idxs(X_train_words, Y_train_words, X_test_words, Y_test_words, max_seqlen)
            return X_train, Y_train, mask_train, X_test, Y_test, mask_test, len(word2idx), max_seqlen, idx2word

        elif self.type == "babi_medium":
            X_train_words, Y_train_words, max_seqlen = self._gen_babi_medium_vecs(self.filename_train, max_seqlen=0)
            X_test_words, Y_test_words, max_seqlen = self._gen_babi_medium_vecs(self.filename_test, max_seqlen)
            X_train, Y_train, mask_train, X_test, Y_test, mask_test, word2idx, idx2word = self._gen_idxs(X_train_words, Y_train_words, X_test_words, Y_test_words, max_seqlen)
            return X_train, Y_train, mask_train, X_test, Y_test, mask_test, len(word2idx), max_seqlen, idx2word

        elif self.type == "babi_full":
            X_train_words, Y_train_words, max_seqlen = self._gen_babi_full_vecs(self.filename_train, max_seqlen=0)
            X_test_words, Y_test_words, max_seqlen = self._gen_babi_medium_vecs(self.filename_test, max_seqlen)
            X_train, Y_train, mask_train, X_test, Y_test, mask_test, word2idx, idx2word = self._gen_idxs(X_train_words, Y_train_words, X_test_words, Y_test_words, max_seqlen)
            return X_train, Y_train, mask_train, X_test, Y_test, mask_test, len(word2idx), max_seqlen, idx2word

    def idx_sentence_to_string(self, x, idx2word):
        cur_sentence = ""
        for idx in x:
            if np.max(idx) > 0:
                cur_sentence += " " + idx2word[np.argmax(idx)]
        return cur_sentence

    def _gen_idxs(self, X_train_words, Y_train_words, X_test_words, Y_test_words, max_seqlen):

        X_train, Y_train, mask_train, X_test, Y_test, mask_test = [], [], [], [], [], []
        word2idx, idx2word, cur_idx = {}, {}, 0

        for sent in X_train_words + X_test_words:
            for word in sent:
                if word not in word2idx:
                    word2idx[word] = cur_idx
                    idx2word[cur_idx] = word
                    cur_idx += 1


        for sentence in X_train_words:
            cur_mask = np.zeros((max_seqlen))
            cur_mask[0:len(sentence)] = 1
            mask_train.append(cur_mask)
            col_idxs = np.asarray([word2idx[w] for w in sentence])
            row_idxs = np.linspace(0, len(sentence) - 1, len(sentence))
            vals = np.ones((len(sentence)))
            cur_sentence_mat = sp.csr_matrix((vals, (row_idxs, col_idxs)), shape=(max_seqlen, cur_idx))
            X_train.append(cur_sentence_mat)

        for sentence in X_test_words:
            cur_mask = np.zeros((max_seqlen))
            cur_mask[0:len(sentence)] = 1
            mask_test.append(cur_mask)
            col_idxs = np.asarray([word2idx[w] for w in sentence])
            row_idxs = np.linspace(0, len(sentence) - 1, len(sentence))
            vals = np.ones((len(sentence)))
            cur_sentence_mat = sp.csr_matrix((vals, (row_idxs, col_idxs)), shape=(max_seqlen, cur_idx))
            X_test.append(cur_sentence_mat)

        for w in Y_train_words:
            cur_y_train = np.zeros((cur_idx))
            cur_y_train[word2idx[w]] = 1
            Y_train.append(cur_y_train)
            #y_train.append(sp.csr_matrix((np.asarray([1]), (np.asarray([0]), np.asarray([word2idx[w]]))), shape=(1, cur_idx)))

        for w in Y_test_words:
            cur_y_test = np.zeros((cur_idx))
            cur_y_test[word2idx[w]] = 1
            Y_test.append(cur_y_test)

        return X_train, Y_train, mask_train, X_test, Y_test, mask_test, word2idx, idx2word

    def _gen_babi_medium_vecs(self, filename, max_seqlen):
        X_words, Y_words = [], []

        with open(filename, encoding='utf-8') as f:
            cur_article = []
            for idx, line in enumerate(f):
                if not idx % self.LEN_BABI_ARTICLE:
                    cur_article = []
                cur_article.append(line.strip())
                if "?" in line.strip():
                    max_seqlen = self._generate_answer_question_pair(line.strip(), cur_article, X_words, Y_words, max_seqlen)
                    max_seqlen = self._gen_medium_babi_pair(cur_article, X_words, idx % self.LEN_BABI_ARTICLE, max_seqlen)

        return X_words, Y_words, max_seqlen

    def _gen_babi_simple_vecs(self, filename, max_seqlen):
        X_words, Y_words = [], []

        with open(filename, encoding='utf-8') as f:
            cur_article = []
            for idx, line in enumerate(f):
                if not idx % self.LEN_BABI_ARTICLE:
                    cur_article = []
                cur_article.append(line.strip())
                if "?" in line.strip():
                    max_seqlen = self._generate_answer_question_pair(line.strip(), cur_article, X_words, Y_words, max_seqlen)

        return X_words, Y_words, max_seqlen

    def _gen_medium_babi_pair(self, cur_article, X_words, cur_position, max_seqlen):
        last_added = X_words[-1]
        added_sentence = None
        if len(cur_article) > 2:
            for idx in range(len(cur_article) - 1, -1, -1):
                if "?" not in cur_article[idx] and last_added[0] not in cur_article[idx]:
                    added_sentence = cur_article[idx].split()[1:]
                    break
        if added_sentence is not None:
            if random.random() > 0.5:
                X_words[-1] = X_words[-1][0:-3] + added_sentence + X_words[-1][-3:]
            else:
                X_words[-1] = added_sentence + X_words[-1]
        if len(X_words[-1]) > max_seqlen:
            max_seqlen = len(X_words[-1])
        return max_seqlen

    def _gen_babi_full_vecs(self, filename, max_seqlen):
        X_words, Y_words = [], []

        with open(filename, encoding='utf-8') as f:
            cur_article = []
            for idx, line in enumerate(f):
                if not idx % self.LEN_BABI_ARTICLE:
                    cur_article = []
                cur_article.append(line.strip())
                if "?" in line.strip():
                    max_seqlen = self._gen_full_article_answer(line.strip(), cur_article, X_words, Y_words, max_seqlen)
        return X_words, Y_words, max_seqlen

    def _gen_full_article_answer(self, question, article, X_words, Y_words, max_seqlen):
        tokenizer = RegexpTokenizer(r'\w+')
        answer =  re.split(r'\t+', question)[1]
        question_txt = tokenizer.tokenize(question)[1:-2]
        article_data = []
        for idx in range(len(article) - 2, -1, -1):
            if "?" not in article[idx]:
                article_data += article[idx].split()[1:]
        article_data += question_txt
        X_words.append(article_data)
        Y_words.append(answer)
        if len(article_data) > max_seqlen:
            max_seqlen = len(article_data)
        return max_seqlen

    def _generate_answer_question_pair(self, question, article, X_train_words, Y_train_words, max_seqlen):

        tokenizer = RegexpTokenizer(r'\w+')
        answer =  re.split(r'\t+', question)[1]
        question_txt = tokenizer.tokenize(question)[1:-2]
        ref = int(re.split(r'\t+', question)[-1]) - 1
        seq = tokenizer.tokenize(article[ref])[1:] + question_txt

        if len(seq) > max_seqlen:
            max_seqlen = len(seq)
        X_train_words.append(seq)
        Y_train_words.append(answer)
        return max_seqlen