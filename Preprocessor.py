import numpy as np
from scipy import sparse as sp
import re, string

class Preprocessor(object):

    def __init__(self, directory, filename_train, filename_test, data_type):
        self.directory = directory
        self.filename_train = filename_train
        self.filename_test = filename_test
        self.data_type = data_type

    def extract_data(self):
        X_train_words, y_train_words, max_seq_len_train = self._get_babi_vecs(self.directory + self.filename_train)
        X_test_words, y_test_words, max_seq_len_test = self._get_babi_vecs(self.directory + self.filename_test)
        max_seq_len = max(max_seq_len_train, max_seq_len_test)

        mask_train, mask_test = self._generate_mask(max_seq_len, X_train_words, X_test_words)
        X_train, y_train, X_test, y_test, idx2word = self._generate_one_hot(X_train_words, y_train_words, X_test_words, y_test_words, max_seq_len)

        return X_train, y_train, mask_train, X_test, y_test, mask_test, X_train[0][0].shape[1], max_seq_len, idx2word

    def _generate_mask(self, max_seq_len, X_train_words, X_test_words):

        mask_train = []
        for x in X_train_words:
            cur_mask = np.zeros((max_seq_len))
            cur_mask[0:len(x)] = 1
            mask_train.append(cur_mask)

        mask_test = []
        for x in X_test_words:
            cur_mask = np.zeros((max_seq_len))
            cur_mask[0:len(x)] = 1
            mask_test.append(cur_mask)

        return mask_train, mask_test

    def _generate_one_hot(self, X_train_words, y_train_words, X_test_words, y_test_words, max_seq_len):

        word2idx = {}
        idx2word = {}

        cur_idx = 0
        for sentence in X_train_words + X_test_words:
            for w in sentence:
                if w not in word2idx:
                    word2idx[w] = cur_idx
                    idx2word[cur_idx] = w
                    cur_idx += 1

        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for sentence in X_train_words:
            col_idxs = np.asarray([word2idx[w] for w in sentence])
            row_idxs = np.linspace(0, len(sentence) - 1, len(sentence))
            vals = np.ones((len(sentence)))
            cur_sentence_mat = sp.csr_matrix((vals, (row_idxs, col_idxs)), shape=(max_seq_len, cur_idx))
            X_train.append(cur_sentence_mat)

        for sentence in X_test_words:
            col_idxs = np.asarray([word2idx[w] for w in sentence])
            row_idxs = np.linspace(0, len(sentence) - 1, len(sentence))
            vals = np.ones((len(sentence)))
            cur_sentence_mat = sp.csr_matrix((vals, (row_idxs, col_idxs)), shape=(max_seq_len, cur_idx))
            X_test.append(cur_sentence_mat)

        for w in y_train_words:
            cur_y_train = np.zeros((cur_idx))
            cur_y_train[word2idx[w]] = 1
            y_train.append(cur_y_train)
            #y_train.append(sp.csr_matrix((np.asarray([1]), (np.asarray([0]), np.asarray([word2idx[w]]))), shape=(1, cur_idx)))

        for w in y_test_words:
            cur_y_test = np.zeros((cur_idx))
            cur_y_test[word2idx[w]] = 1
            y_test.append(cur_y_test)
            #y_test.append(sp.csr_matrix((np.asarray([1]), (np.asarray([0]), np.asarray([word2idx[w]]))), shape=(1, cur_idx)))

        return X_train, y_train, X_test, y_test, idx2word

    def _get_babi_vecs(self, filename):
        pattern = re.compile('[\W_]+')
        X = []
        y = []
        max_seq_len = 0
        with open(filename, 'r') as f:
            content = f.readlines()
        for idx in range(0, len(content), 3):
            cur_example = []
            for w in content[idx].split() + content[idx + 1].split():
                if pattern.sub('', w).isalpha():
                    cur_example.append(pattern.sub('', w))
            question = content[idx + 2].split()[0:-2]
            question.pop(0)
            [cur_example.append(pattern.sub('', w)) for w in question]
            answer = content[idx + 2].split()[-2]
            if len(cur_example) > max_seq_len:
                max_seq_len = len(cur_example)
            X.append(cur_example)
            y.append(answer)

        return X, y, max_seq_len


