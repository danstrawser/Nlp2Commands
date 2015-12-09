import os
from nltk.tokenize import sent_tokenize
import nltk
import re
import random

class CNNProcessor(object):

    def __init__(self, type_of_data):
        self.directory = 'data/rc-data/data/cnn_2/questions'
        self.training_directory = 'training_improved'
        self.test_directory = 'test'
        assert(type_of_data == "full" or type_of_data == "simplified" or type_of_data == "medium")
        self.type_of_data = type_of_data
        self.MAX_NUM_TRAINING_FILES = 20000
        self.MAX_NUM_TEST_FILES = 500

    def process(self):
        test_lines, train_lines, max_seqlen = [], [], 0

        for idx, file in enumerate(os.listdir(self.directory + '/' + self.training_directory)):
            with open(self.directory + '/' + self.training_directory + '/' + file, encoding="utf8") as f:
                cur_article = []
                [cur_article.append(line.strip()) for line in f if line.strip()]
                if self.type_of_data == "full":
                    max_seqlen = self._generate_line(train_lines, cur_article[1], cur_article[2], cur_article[3], max_seqlen)
                elif self.type_of_data == "simplified":
                    max_seqlen = self._generate_simplified_line(train_lines, cur_article[1], cur_article[2], cur_article[3], max_seqlen)
                elif self.type_of_data == "medium":
                    max_seqlen = self._generate_medium_line(train_lines, cur_article[1], cur_article[2], cur_article[3], max_seqlen)
            if idx > self.MAX_NUM_TRAINING_FILES:
                break

        for idx, file in enumerate(os.listdir(self.directory + '/' + self.test_directory)):
            with open(self.directory + '/' + self.test_directory + '/' + file, encoding="utf8") as f:
                cur_article = []
                [cur_article.append(line.strip()) for line in f if line.strip()]
                if self.type_of_data == "full":
                    max_seqlen = self._generate_line(test_lines, cur_article[1], cur_article[2], cur_article[3], max_seqlen)
                elif self.type_of_data == "simplified":
                    max_seqlen = self._generate_simplified_line(test_lines, cur_article[1], cur_article[2], cur_article[3], max_seqlen)
                elif self.type_of_data == "medium":
                    max_seqlen = self._generate_medium_line(test_lines, cur_article[1], cur_article[2], cur_article[3], max_seqlen)

            if idx > self.MAX_NUM_TEST_FILES:
                break

        word2idx, idx2word, vocab, max_sent_len = self._word2idx(train_lines + test_lines)
        #train_lines, test_lines = self._split_test_train(lines)
        return vocab, train_lines, test_lines, word2idx, idx2word, max_seqlen, max_sent_len


    def _generate_medium_line(self, lines, article_text, question, answer, max_seqlen):
        probability_accepting_new_sentence = 0.3
        cur_idx, seqlen = 1, 0
        for s in sent_tokenize(article_text, 'english'):
            if answer in s:
                lines.append({'text': re.sub("@", '', s), 'type': 's'})
                cur_idx += 1
                seqlen += 1
            if answer not in s:
                if random.random() < probability_accepting_new_sentence:
                    lines.append({'text': re.sub("@", '', s), 'type': 's'})
                    cur_idx += 1
                    seqlen += 1

        if seqlen > max_seqlen:
            max_seqlen = seqlen
        lines.append({'answer': re.sub("@", '', answer), 'text': re.sub("@", '', question), 'refs': [1], 'id': cur_idx, 'type': 'q'})

    def _generate_simplified_line(self, lines, article_text, question, answer, max_seqlen):
        cur_idx, seqlen = 1, 0
        for s in sent_tokenize(article_text, 'english'):
            if answer in s:
                lines.append({'text': re.sub("@", '', s), 'type': 's'})
                cur_idx += 1
                seqlen += 1
        if seqlen > max_seqlen:
            max_seqlen = seqlen
        lines.append({'answer': re.sub("@", '', answer), 'text': re.sub("@", '', question), 'refs': [1], 'id': cur_idx, 'type': 'q'})
        return max_seqlen

    def _generate_line(self, lines, article_text, question, answer, max_seqlen):
        cur_idx, cur_seq_len = 0, 0
        for s in sent_tokenize(article_text, 'english'):
            lines.append({'text': s, 'type': 's'})
            cur_idx += 1
            cur_seq_len += 1
        if cur_seq_len > max_seqlen:
            max_seqlen = cur_seq_len
        lines.append({'answer': answer, 'text': question, 'refs': [1], 'id': cur_idx, 'type': 'q'})
        return max_seqlen

    def _word2idx(self, lines):
        word2idx, idx2word, vocab, cur_idx, max_sent_len = {}, {}, set(), 0, 0

        for l in lines:
            sent = nltk.word_tokenize(l['text'])
            #print(" sent: ", sent)
            if len(sent) > max_sent_len:
                max_sent_len = len(sent)
            for idx, w in enumerate(sent):
                if w not in word2idx:
                    word2idx[w] = cur_idx
                    idx2word[cur_idx] = w
                    vocab.add(w)
                    cur_idx += 1

        #assert(1 ==2 )
        return word2idx, idx2word, vocab, max_sent_len

    def _split_test_train(self, lines):
        percent_train = 0.9
        idx_train = percent_train * len(lines)
        train_lines = []
        test_lines = []
        started_test_lines = 0
        last_type = 's'
        for idx, l in enumerate(lines):
            if not started_test_lines and idx >= idx_train and last_type == 'q' and l['type'] == 's':
                started_test_lines = 1
            if started_test_lines:
                test_lines.append(l)
            else:
                train_lines.append(l)
            last_type = l['type']

        return train_lines, test_lines
