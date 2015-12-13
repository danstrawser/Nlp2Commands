__author__ = 'Dan'

from nltk.tokenize import RegexpTokenizer
import numpy as np
import re
from copy import deepcopy

class babi_processor_dmn(object):

    def __init__(self, filename_train, filename_test):
        self.filename_train = filename_train
        self.filename_test = filename_test

    def process(self):

        self.babi_article_len = 15
        self.tokenizer = RegexpTokenizer(r'\w+')

        X_train, mask_words_train, mask_article_train, Question_train, Question_mask_train, Y_train = [], [], [], [], [], []
        X_test, mask_words_test, mask_article_test, Question_test, Question_mask_test, Y_test = [], [], [], [], [], []
        max_sentlen, max_sentences_per_article, max_question_len = 0, 0, 0
        word2idx, idx2word = {}, {}

        X_train, Question_train, Answer_train = self._gen_word_vecs(self.filename_train)
        X_test, Question_test, Answer_test = self._gen_word_vecs(self.filename_test)

        cur_idx = 0
        word2idx["<BLANK>"] = cur_idx
        idx2word[cur_idx] = "<BLANK>"
        cur_idx += 1
        word2idx["<EOS>"] = cur_idx
        idx2word[cur_idx] = "<EOS>"
        cur_idx += 1
        for article in X_train + X_test:
            if len(article) > max_sentences_per_article:
                max_sentences_per_article = len(article)
            for sent in article:
                if len(sent) > max_sentlen:
                    max_sentlen = len(sent)
                for w in sent:
                    if w not in word2idx:
                        word2idx[w] = cur_idx
                        idx2word[cur_idx] = w
                        cur_idx += 1

        for sent in Question_train + Question_test:
            if len(sent) > max_question_len:
                max_question_len = len(sent)
            for w in sent:
                if w not in word2idx:
                    word2idx[w] = cur_idx
                    idx2word[cur_idx] = w
                    cur_idx += 1

        X_vec_train, sentence_mask_train, article_mask_train, Question_vec_train, Answer_vec_train = self._gen_idx_vecs(X_train, Question_train, Answer_train, word2idx, max_sentences_per_article, max_sentlen, max_question_len)
        X_vec_test, sentence_mask_test, article_mask_test, Question_vec_test, Answer_vec_test = self._gen_idx_vecs(X_test, Question_test, Answer_test, word2idx, max_sentences_per_article, max_sentlen, max_question_len)

        return X_vec_train, sentence_mask_train, Question_vec_train, Answer_vec_train, X_vec_test, sentence_mask_test, Question_vec_test, Answer_vec_test

    def _gen_idx_vecs(self, X, Question, Answer, word2idx, max_sentences_per_article, max_sentlen, max_question_len):
        X_vec, sentence_mask, article_mask, Question_vec, Answer_vec = [], [], [], [], []

        X_vec = word2idx["<BLANK>"] * np.ones((len(X), max_sentences_per_article, max_sentlen))
        Question_vec = word2idx["<BLANK>"] * np.ones((len(X), max_question_len))
        sentence_mask = np.zeros((len(X), max_sentences_per_article, max_sentlen))
        article_mask = np.zeros((len(X), max_sentences_per_article))

        for idx, article in enumerate(X):
            article_mask[idx, :len(article)] = 1
            for jdx, sent in enumerate(article):
                sentence_mask[idx, jdx, :len(sent)] = 1
                for kdx, w in enumerate(sent):
                    X_vec[idx, jdx, kdx] = word2idx[w]
        for idx, question in enumerate(Question):
            for jdx, w in enumerate(question):
                Question_vec[idx, jdx] = word2idx[w]
        for a in Answer:
            Answer_vec.append(word2idx[a])

        return  X_vec, sentence_mask, article_mask, Question_vec, Answer_vec

    def _gen_word_vecs(self, filename):
        X, Question, Answer = [], [], []
        with open(filename, encoding='utf-8') as f:
            cur_article = []
            cur_article_idx = 0
            for line in f:
                cur_article_idx += 1
                if "?" not in line:
                    cur_article.append(self.tokenizer.tokenize(line.strip())[1:])
                else:
                    X.append(deepcopy(cur_article))
                    ques_answer = re.split(r'\t+', line.strip())
                    Question.append(self.tokenizer.tokenize(ques_answer[0])[1:])
                    Answer.append(ques_answer[1])
                if cur_article_idx % self.babi_article_len == 0:
                    cur_article = []
        return X, Question, Answer



















    def process_data(self, type_of_embedding):

        filename_train = 'babi_train1.txt'
        filename_test = 'babi_test1.txt'

        X_train, mask_sentences_train, mask_articles_train, Question_train, Question_train_mask, Y_train, X_test, mask_sentences_test, mask_articles_test, Question_test, Question_test_mask, Y_test, max_queslen = [], [], [], [], [], [], [], [], [], [], [], [], 0

        fact_ordering_train, fact_ordering_test = [], []

        cur_idx = 1
        word2idx = {}
        idx2word = {}
        tokenizer = RegexpTokenizer(r'\w+')

        word2idx["<BLANK>"] = 0
        idx2word[0] = "<BLANK>"

        max_article_len = 0
        max_sentence_len = 0
        max_queslen = 0

        with open(filename_train, encoding='utf-8') as f:
            cur_article = []
            for line in f:
                if "?" not in line and "@" not in line and "$" not in line:
                    cur_sentence = []
                    for w in tokenizer.tokenize(line):
                        cur_sentence.append(w.strip().lower())
                    cur_article.append(cur_sentence)
                    if len(cur_sentence) > max_sentence_len:
                        max_sentence_len = len(cur_sentence)
                else:
                    fact_ordering_train.append(np.asarray([int(line[2:])], dtype='int32'))

                    question_phrase = next(f)[2:].split()
                    cur_question = []

                    if len(cur_article) > max_article_len:
                        max_article_len = len(cur_article)
                    if len(question_phrase) > max_queslen:
                        max_queslen = len(question_phrase)
                    for w in question_phrase:
                        cur_question.append(w.strip().lower())

                    Question_train.append((cur_question))
                    X_train.append(cur_article)
                    Y_train.append(next(f)[2:].strip())
                    cur_article = []

        with open(filename_test, encoding='utf-8') as f:
            cur_article = []
            for line in f:
                if "?" not in line and "@" not in line and "$" not in line:
                    cur_sentence = []
                    for w in tokenizer.tokenize(line):
                        cur_sentence.append(w.strip().lower())
                    cur_article.append(cur_sentence)
                    if len(cur_sentence) > max_sentence_len:
                        max_sentence_len = len(cur_sentence)
                else:
                    fact_ordering_test.append(np.asarray([int(line[2:])], dtype='int32'))

                    question_phrase = next(f)[2:].split()
                    cur_question = []
                    if len(cur_article) > max_article_len:
                        max_article_len = len(cur_article)
                    if len(question_phrase) > max_queslen:
                        max_queslen = len(question_phrase)
                    for w in question_phrase:
                        cur_question.append(w.strip().lower())
                    Question_test.append(cur_question)
                    X_test.append(cur_article)
                    Y_test.append(next(f)[2:].strip())
                    cur_article = []

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
                for w in s:
                    if w not in word2idx:
                        word2idx[w] = cur_idx
                        idx2word[cur_idx] = w
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

        word2idx["<EOS>"] = cur_idx
        idx2word[cur_idx] = "<EOS>"
        cur_idx += 1

        X_train_vec, Question_train_vec, Y_train_vec, X_test_vec, Question_test_vec, Y_test_vec = [], [], [], [], [], []

        for article in X_train:
            cur_article = np.zeros((max_article_len, max_sentence_len), dtype='int32')
            for idx, fact in enumerate(article):
                for jdx, word in enumerate(fact):
                    cur_article[idx][jdx] = word2idx[word]
            X_train_vec.append(np.asarray(cur_article))

        for article in X_test:
            cur_article = np.zeros((max_article_len, max_sentence_len), dtype='int32')
            for idx, fact in enumerate(article):
                for jdx, word in enumerate(fact):
                    cur_article[idx][jdx] = word2idx[word]
            X_test_vec.append(np.asarray(cur_article))

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
                    new_vec = word2idx[w]
                else:
                    assert(1 == 2) # Not implemented
                    #new_vec = np.zeros((len(word2idx)))
                    #new_vec[word2idx[f]] = 1
                cur_question.append(new_vec)
            Question_train_vec.append((cur_question))


        for q in Question_test:
            cur_question = []
            for w in q:
                if type_of_embedding == "embeddings":
                    new_vec = word2idx[w]
                else:
                    assert(1 == 2) # Not implemented
                    #new_vec = np.zeros((len(word2idx)))
                    #new_vec[word2idx[f]] = 1
                cur_question.append(new_vec)
            Question_test_vec.append((cur_question))

        # Ensure we have all the same length
        new_question_train_vec = []
        for q in Question_train_vec:
            for added_el in range(len(q), max_queslen):
                q = np.concatenate((q, [0]), axis=0)
            new_question_train_vec.append(q)
        Question_train_vec = new_question_train_vec

        new_question_test_vec = []
        for q in Question_test_vec:
            for added_el in range(len(q), max_queslen):
                q = np.concatenate((q, [0]), axis=0)
            new_question_test_vec.append(q)
        Question_test_vec = new_question_test_vec


        assert(len(X_test_vec) == len(Y_test_vec))

        # TODO: What needs edited:  the x_train_vec, x_test_vec, x_mask
        total_sequence_length = max_sentence_len * max_article_len + max_article_len
        max_sentence_len = max_sentence_len +1

        for article in X_train:
            cur_mask_sentence = np.zeros(0, dtype='int32')
            for sentence in article:
                cur_mask_sentence_new = np.zeros(max_sentence_len, dtype='int32')
                cur_mask_sentence_new[0:len(sentence)] = 1
                cur_mask_sentence_new[-1] = 1
                cur_mask_sentence = np.concatenate((cur_mask_sentence, cur_mask_sentence_new), axis=0)
            if len(cur_mask_sentence) < total_sequence_length:
                cur_mask_sentence = np.concatenate((cur_mask_sentence, [0, 0, 0, 0, 0, 0, 1] ), axis=0)
            mask_sentences_train.append(cur_mask_sentence)

        for article in X_test:
            cur_mask_sentence = np.zeros(0, dtype='int32')
            for sentence in article:
                cur_mask_sentence_new = np.zeros(max_sentence_len, dtype='int32')
                cur_mask_sentence_new[0:len(sentence)] = 1
                cur_mask_sentence_new[-1] = 1
                cur_mask_sentence = np.concatenate((cur_mask_sentence, cur_mask_sentence_new), axis=0)
            if len(cur_mask_sentence) < total_sequence_length:
                cur_mask_sentence = np.concatenate((cur_mask_sentence, [0, 0, 0, 0, 0, 0, 1] ), axis=0)
            mask_sentences_test.append(cur_mask_sentence)

        #X_train_vec_new, X_test_vec_new = [], []
        X_train_vec_new = np.zeros((len(X_train_vec), total_sequence_length),dtype='int32')
        X_test_vec_new = np.zeros((len(X_train_vec), total_sequence_length),dtype='int32')

        for idx, x in enumerate(X_train_vec):
            cur_sentence = np.zeros(total_sequence_length, dtype='int32')
            for jdx, s in enumerate(x):
                res =  [word2idx["<EOS>"]]
                cur_sentence[jdx * (max_sentence_len): (jdx + 1) * (max_sentence_len)] = np.concatenate((s, res), axis=1)
            X_train_vec_new[idx, :] = cur_sentence

        for idx, x in enumerate(X_test_vec):
            cur_sentence = np.zeros(total_sequence_length, dtype='int32')
            for jdx, s in enumerate(x):
                res =  [word2idx["<EOS>"]]
                cur_sentence[jdx * (max_sentence_len): (jdx + 1) * (max_sentence_len)] = np.concatenate((s, res), axis=1)
            X_test_vec_new[idx, :] = cur_sentence

        X_train_vec = X_train_vec_new
        X_test_vec = X_test_vec_new

        assert(len(X_train_vec[0]) == total_sequence_length)
        assert(total_sequence_length == max_article_len * max_sentence_len)


        return X_train_vec, mask_sentences_train, fact_ordering_train, Question_train_vec, Question_train_mask, Y_train_vec, X_test_vec, mask_sentences_test, fact_ordering_test, Question_test_vec, Question_test_mask, Y_test_vec, word2idx, idx2word, len(word2idx), max_queslen, max_sentence_len, total_sequence_length

