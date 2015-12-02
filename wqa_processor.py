import os
import re
import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize import RegexpTokenizer
import nltk
import codecs

MAX_SENT_LEN = 100

class WikiProcessor(object):

    def __init__(self, filename):
        self.filename = filename

    def process(self, extract_yes_no=1):

        docs = {}

        nums = ['S08','S09','S10']
        question_filename = "question_answer_pairs.txt"
        max_file_num = 10

        print("processing")
        for dir in nums:
            cur_directory = 1

            while os.path.exists(self.filename + dir + "/data/set" + str(cur_directory) + "/"):

                for file_num in range(max_file_num):
                    article = []
                    with open(self.filename + dir + "/data/set" + str(cur_directory) + "/" + "a" + str(file_num + 1) + ".txt.clean", encoding='utf-8') as f:
                        [article.append(line.strip()) for line in f if line.strip()]
                    title = article[0].replace(" ","_")
                    docs[title] = {}
                    docs[title]['article'] = article[1:]
                    docs[title]['questions'] = {}
                cur_directory += 1

        for dir in nums:
            questions = []
            with open(self.filename + dir + "/" + question_filename,encoding='utf-8') as f:
                for line in f:
                    questions.append(re.split(r'\t+', line))
            for q in questions[1:]:
                docs[q[0]]['questions'][q[1]] = {'answer': -1, 'difficulty': -1}
                docs[q[0]]['questions'][q[1]]['answer'] = q[2] #.translate(string.maketrans("",""), string.punctuation)
                docs[q[0]]['questions'][q[1]]['difficulty'] = q[3]

        if extract_yes_no:
            self._extract_yes_or_no(docs)

        lines, max_seqlen = self._convert_to_mem_net_format(docs)
        lines, vocab, word2idx, idx2word, max_sent_len = self._word2idx(lines, docs)
        test_lines, train_lines = self._split_test_train(lines)

        return vocab, train_lines, test_lines, word2idx, idx2word, max_seqlen, max_sent_len

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

    def _convert_to_mem_net_format(self, docs):
        #tokenizer = RegexpTokenizer(r'\w+')
        # print(" w token ;", tokenizer.tokenize(s).lower())
        max_seqlen = 0
        lines = []
        for d in docs.keys():
            cur_line = 0
            for p in docs[d]['article']:
                #p = p.decode('utf-8')
                for s in sent_tokenize(p, 'english'):
                    lines.append({'text': s, 'type': 's'})
                    cur_line += 1

            for q in docs[d]['questions'].keys():
                a = docs[d]['questions'][q]
                lines.append({'answer': a['answer'], 'text': q, 'type': 'q', 'refs': [0], 'id': cur_line})
                cur_line += 1
            if cur_line > max_seqlen:
                max_seqlen = cur_line

        return lines, max_seqlen

    def _extract_yes_or_no(self, docs):
        for d in docs.keys():
            updated_questions = {}
            for q in docs[d]['questions'].keys():
                dic = docs[d]['questions'][q]
                if dic['answer'].lower() == 'yes' or dic['answer'].lower() == 'no':
                    updated_questions[q] = {'answer':dic['answer'], 'difficulty':dic['difficulty']}
            docs[d]['questions'] = updated_questions


    def _word2idx(self, lines, docs):
        word2idx, idx2word = {}, {}
        cur_idx, max_sent_len, max_seq_len = 0, 0, 0
        vocab = set()

        cur_idx = 0
        for p in docs.keys():
            if len(docs[p]['article']) > max_seq_len:
                max_seq_len = len(docs[p]['article'])

        new_lines = []
        for l in lines:
            sent = nltk.word_tokenize(l['text'])
            for w in sent:
                if w not in word2idx:
                    vocab.add(w)
                    word2idx[w] = cur_idx
                    idx2word[cur_idx] = w
                    cur_idx += 1

            if len(sent) < MAX_SENT_LEN:
                cur_l = l
                new_lines.append(cur_l)

        return new_lines, vocab, word2idx, idx2word, MAX_SENT_LEN