from theano.gradient import disconnected_grad
__author__ = 'Dan'

from theano.compile.mode import FAST_COMPILE
from numpy import dtype
import numpy as np
import re
import theano
import theano.tensor as T
import theano.typed_list
from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
import random
from random import shuffle
import pickle
import sys
import os
from theano import config

class DMN_SimplerGate(object):

    # We take as input a string of "facts"
    def __init__(self, num_fact_hidden_units, number_word_classes, dimension_fact_embeddings, num_episode_hidden_units, max_number_of_facts_read):

        dimension_fact_embeddings = 8
        print(" Starting dmn no scan... ")
        self.preprocess_babi_set_for_dmn()

        print(" Starting dmn no scidxsan... ")
        
        self.X_train, self.mask_sentences_train, self.fact_ordering_train, self.question_train, self.question_train_mask, self.Y_train, self.X_test, self.mask_sentences_test, self.fact_ordering_test, self.question_test, self.question_test_mask, self.Y_test, word2idx, self.idx2word, dimension_fact_embeddings, max_queslen, max_sentlen, total_sequence_length = self.process_data("embeddings")
               
               
        print(" Building model... ")
        number_word_classes = max(self.idx2word.keys(), key=int) + 1
        #max_fact_seqlen = max_article_len
        max_fact_seqlen = max_sentlen
        
        total_number_of_sentences_per_episode = int(total_sequence_length / max_fact_seqlen)
        max_number_of_facts_read = 1
        dimension_word_embeddings = 10
        max_number_of_episodes_read = 1
        self.initialization_randomization = 1

        nh = 8 # Dimension of the hidden layer
        num_hidden_units = nh
        num_hidden_units_facts = num_hidden_units
        num_hidden_units_episodes = num_hidden_units_facts
        num_hidden_units_questions = num_hidden_units_episodes
        num_hidden_units_words = num_hidden_units_questions

        total_len_word_seq = total_sequence_length
                
        self.initialize_dmn_params(nh, num_hidden_units_words, num_hidden_units_facts, num_hidden_units_episodes, num_hidden_units_questions, dimension_word_embeddings, dimension_fact_embeddings, max_fact_seqlen, max_number_of_episodes_read, number_word_classes, total_number_of_sentences_per_episode)
        
        question_encoding = self.GRU_question(dimension_fact_embeddings, num_hidden_units_questions, num_hidden_units_episodes, max_queslen, dimension_word_embeddings)

        # Size is (num_episodes_read, word_idx_in_sentence)
        word_idxs = T.lmatrix("word_indices") # as many columns as words in the context window and as many lines as words in the sentence
        word_mask = T.lmatrix("word_mask")
        facts_input = T.lvector("fact_indices")

        #sentence_mask = T.lvector("sentence_mask")
        y_sentence = T.lscalar('y_sentence')

        
        def slice_w(x, n):
            return x[n*num_hidden_units_facts:(n+1)*num_hidden_units_facts]

        def gru_reader_step(x_cur_word, h_prev, w_mask):

            W_in_stacked = T.concatenate([self.W_word_reset_gate_x, self.W_word_update_gate_x, self.W_word_hidden_gate_x], axis=1)  # I think your issue is that this should have # dim word embeddings
            W_hid_stacked = T.concatenate([self.W_word_reset_gate_h, self.W_word_update_gate_h, self.W_word_hidden_gate_h], axis=1)

            input_n = T.dot(x_cur_word, W_in_stacked)
            hid_input = T.dot(h_prev, W_hid_stacked)

            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = T.tanh(resetgate)
            updategate = T.tanh(updategate)

            hidden_update = slice_w(input_n, 2) + resetgate * slice_w(hid_input, 2)
            hidden_update = T.tanh(hidden_update)
            h_cur = (1 - updategate) * hidden_update + updategate * hidden_update

            h_cur = w_mask * h_cur + (1 - w_mask) * h_prev
            # h_cur = T.tanh(T.dot(self.W_fact_to_hidden, x_cur) + T.dot(self.W_hidden_to_hidden, h_prev))
            return h_cur

        list_of_fact_softmaxes = []

        def brain_gru_reader(episode_reading_idx, h_prev_episode):

            h_cur = question_encoding # Initial state of word GRU
            current_facts = None

            for idx in range(total_len_word_seq):

                h_cur = gru_reader_step(self.emb[word_idxs[episode_reading_idx][idx]], h_cur, word_mask[episode_reading_idx][idx])

                if idx % max_fact_seqlen == 0:
                    output = T.tanh(T.dot(self.W_word_to_fact_vector, h_cur) + self.b_word_to_fact_vector)

                    if current_facts is None:
                        current_facts = [output]  # Run through softmax
                    else:
                        current_facts = T.concatenate((current_facts, [output]), axis=0)
                        
            z_dmn = T.concatenate(([question_encoding], [h_prev_episode]), axis=0)  # This has dimension (2,8)
            
            self.G_dmn = T.nnet.sigmoid(T.dot(self.W_dmn_2, T.tanh(T.dot(z_dmn, self.W_dmn_1)) + self.b_dmn_1) + self.b_dmn_2)  # Note that this should be 1-dimensional
            self.cur_facts = current_facts
            
            # Current_facts is (2, 8).   You want the output to be (2,1).  This will necessitate:
            # G_dmn has size (8,1)
            
            self.t_dot_res = T.dot(current_facts, self.G_dmn).T
            result_of_gate = T.nnet.softmax(self.t_dot_res).T  # Ensure that this has 1 dimension, should be weight of whether or not we take g_i * F_i
            
            # Note, I think result of gate is (1,2)            
            list_of_fact_softmaxes.append(result_of_gate)

            # Believe error is here 
            brain_input = T.dot(current_facts.T, result_of_gate)  # I believe this has size (2, 8)
            # I believe brian is 8 x 1

            # W_in_stacked is 8x63, each individual element is (8,21)
            W_in_stacked = T.concatenate([self.W_episode_reset_gate_x, self.W_episode_update_gate_x, self.W_episode_hidden_gate_x], axis=1)  # I think your issue is that this should have # dim brain embeddings
            W_hid_stacked = T.concatenate([self.W_episode_reset_gate_h, self.W_episode_update_gate_h, self.W_episode_hidden_gate_h], axis=1)
            # W_in_stacked is 8x24

            # brain_input is DEFINITELY (8,1)  
            input_n = T.dot(brain_input.T, W_in_stacked)  #  Line is fine
            hid_input = T.dot(h_prev_episode, W_hid_stacked)  # This is not the error

            # LAST RUN THE SECOND ONE HAD TOO MANY!
            # Now the first one has too many, definitely the following line
            
            # Note:  input_in has size (1,24).  Note but there is an extra dimension 
            self.output_in = input_n
            
            resetgate = slice_w(input_n[0], 0) + slice_w(hid_input, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n[0], 1) # Just this line is problem 
            resetgate = T.tanh(resetgate)
            updategate = T.tanh(updategate)

            hidden_update = resetgate * slice_w(hid_input, 2) + slice_w(input_n[0], 2)
            hidden_update = T.tanh(hidden_update)
            h_cur = (1 - updategate) * hidden_update + updategate * hidden_update

            # h_cur = T.tanh(T.dot(self.W_fact_to_hidden, x_cur) + T.dot(self.W_hidden_to_hidden, h_prev))
            return h_cur

        # Brain for loop
        state_episode_step = question_encoding
        # Main Brain Loop
        for idx in range(max_number_of_facts_read):
            state_episode_step = brain_gru_reader(idx, state_episode_step)

        # Episode to word is size: (number_word_classes, num_hidden_units_facts)
        
        output = T.nnet.softmax(T.dot(self.W_episode_to_word, state_episode_step) + self.b_episode_to_word)
        self.out_softmax = output
                
        p_y_given_x_sentence = output[0, :]

        y_pred = T.argmax(p_y_given_x_sentence, axis=0)
        lr = T.scalar('lr')
        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[y_sentence])

        fact_nll = 0        
        for idx in range(max_number_of_facts_read):
            fact_nll += -T.mean(T.log(list_of_fact_softmaxes[idx])[facts_input[idx]])

        fact_grads = T.grad(fact_nll, self.fact_params)
        fact_updates = OrderedDict((p, p - lr*g) for p, g in zip(self.fact_params, fact_grads))  # computes the update for each of the params.

        sentence_gradients = T.grad(sentence_nll, self.params, disconnected_inputs='warn')  # Returns gradients of the nll w.r.t the params
        sentence_updates = OrderedDict((p, p - lr*g) for p, g in zip(self.params, sentence_gradients))  # computes the update for each of the params.

        print("Compiling fcns...")
        self.classify = theano.function(inputs=[word_idxs, word_mask, self.question_idxs, self.question_mask], outputs=y_pred)
                                
        self.sentence_train = theano.function(inputs=[word_idxs, word_mask, self.question_idxs, self.question_mask, y_sentence, lr], outputs=sentence_nll, updates=sentence_updates)
        
        self.out_in = theano.function(inputs=[word_idxs, word_mask, self.question_idxs, self.question_mask, y_sentence, lr], outputs=self.output_in, updates=sentence_updates, on_unused_input='warn')
        self.sm = theano.function(inputs=[word_idxs, word_mask, self.question_idxs, self.question_mask, y_sentence, lr], outputs=self.out_softmax, updates=sentence_updates, on_unused_input='warn')
        
        # The problem is here
        self.fact_train = theano.function(inputs=[word_idxs, word_mask, facts_input, self.question_idxs, self.question_mask, lr], outputs=fact_nll, updates=fact_updates, on_unused_input='warn')
        
        #self.eval_fact_train = theano.function(inputs=[word_idxs, word_mask, facts_input, self.question_idxs, self.question_mask, lr], outputs=list_of_fact_softmaxes[0], updates=fact_updates, on_unused_input='warn')
        #self.get_gdmn = theano.function(inputs=[word_idxs, word_mask, facts_input, self.question_idxs, self.question_mask, lr], outputs=self.G_dmn, updates=fact_updates, on_unused_input='warn')
        #self.get_curf = theano.function(inputs=[word_idxs, word_mask, facts_input, self.question_idxs, self.question_mask, lr], outputs=self.cur_facts, updates=fact_updates, on_unused_input='warn')
        #self.t_dot = theano.function(inputs=[word_idxs, word_mask, facts_input, self.question_idxs, self.question_mask, lr], outputs=self.t_dot_res, updates=fact_updates, on_unused_input='warn')
              
        print("Done compiling!")


    def GRU_question(self, dimension_fact_embedding, num_hidden_units_questions, num_hidden_units_episodes, max_question_len, dimension_word_embeddings):

        self.question_idxs = T.lmatrix("question_indices") # as many columns as words in the context window and as many lines as words in the sentence
        self.question_mask = T.lvector("question_mask")
        q = self.emb[self.question_idxs].reshape((self.question_idxs.shape[0], dimension_word_embeddings)) # x basically represents the embeddings of the words IN the current sentence.  So it is shape

        def slice_w(x, n):
            return x[n*num_hidden_units_questions:(n+1)*num_hidden_units_questions]

        def question_gru_recursion(x_cur, h_prev, q_mask):

            W_in_stacked = T.concatenate([self.W_question_reset_gate_x, self.W_question_update_gate_x, self.W_question_hidden_gate_x], axis=1)
            W_hid_stacked = T.concatenate([self.W_question_reset_gate_h, self.W_question_update_gate_h, self.W_question_hidden_gate_h], axis=1)

            input_n = T.dot(x_cur, W_in_stacked)
            hid_input = T.dot(h_prev, W_hid_stacked)

            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = T.tanh(resetgate)
            updategate = T.tanh(updategate)

            hidden_update = slice_w(input_n, 2) + resetgate * slice_w(hid_input, 2)
            hidden_update = T.tanh(hidden_update)
            h_cur = (1 - updategate) * hidden_update + updategate * hidden_update

            h_cur = q_mask * h_cur + (1 - q_mask) * h_prev
            # h_cur = T.tanh(T.dot(self.W_fact_to_hidden, x_cur) + T.dot(self.W_hidden_to_hidden, h_prev))
            return h_cur

        state = self.h0_questions
        for jdx in range(max_question_len):
            state = question_gru_recursion(q[jdx], state, self.question_mask[jdx])

        return T.tanh(T.dot(state, self.W_question_to_vector) + self.b_question_to_vector)

    def idx2sentence(self, x):
        cur_sent = ""
        for w in x:             
            cur_sent += " " + self.idx2word[int(w)]
        return cur_sent

    def train(self):
        # self.X_train, self.mask_train, self.question_train, self.Y_train, self.X_test, self.mask_test, self.question_test, self.Y_test, word2idx, idx2word, dimension_fact_embeddings = self.process_data()
        lr = .01
        max_epochs = 20000
        print(" Starting training...")
        last_ll = 100000
        min_ll_seen = 10000
        best_correct_seen = 0
        for e in range(max_epochs):
            
            shuffled_idxs = [i for i in range(len(self.X_train))]           
            shuffle(shuffled_idxs)
            ll = 0
            num_train_correct, tot_num_train = 0, 0
            
            ll_fact = 0
            for idx in shuffled_idxs:
              
                x, word_mask, fact_ordering, q, mask_question, y = self.X_train[idx], self.mask_sentences_train[idx], self.fact_ordering_train[idx], self.question_train[idx], self.question_train_mask[idx], self.Y_train[idx]
                ll_fact += self.fact_train([x], [word_mask], fact_ordering, q, mask_question, lr)
                
#                 val_you = self.eval_fact_train([x], [word_mask], fact_ordering, q, mask_question, lr)
#                 print(" one val you: ", val_you)
#                 gdm = self.get_gdmn([x], [word_mask], fact_ordering, q, mask_question, lr)
#                 
#                 print(" this is gdm :" , gdm)
#                 curf = self.get_curf([x], [word_mask], fact_ordering, q, mask_question, lr)
#                 print(" curf: ", curf)
#                 res = self.t_dot([x], [word_mask], fact_ordering, q, mask_question, lr)
#                 
#                 print(" res: ", res)
#             
            print(" This is the ll fact: ", ll_fact)
            
            for idx in shuffled_idxs:
                x, word_mask, fact_ordering, q, mask_question, y = self.X_train[idx], self.mask_sentences_train[idx], self.fact_ordering_train[idx], self.question_train[idx], self.question_train_mask[idx], self.Y_train[idx]
                ll += self.sentence_train([x], [word_mask], q, mask_question, y, lr)
            
                outin = self.out_in([x], [word_mask], q, mask_question, y, lr)
                sm = self.sm([x], [word_mask], q, mask_question, y, lr)
                
                #print(" soft max res: ", sm)
            
            
            shuffle(shuffled_idxs)
            for idx in shuffled_idxs:
                x, word_mask, fact_ordering, q, mask_question, y = self.X_train[idx], self.mask_sentences_train[idx], self.fact_ordering_train[idx], self.question_train[idx], self.question_train_mask[idx], self.Y_train[idx]
                predictions_test = self.classify([x], [word_mask], q, mask_question)
                if predictions_test == y:
                    num_train_correct += 1
                tot_num_train += 1

#               
            print(" ratio training data predicted correctly: ", num_train_correct / tot_num_train)

            correct = 0
            total_tests = 0
            for idx in range(len(self.X_test)):
                x, word_mask, fact_ordering, q, mask_question, y = self.X_test[idx], self.mask_sentences_test[idx], self.fact_ordering_test[idx], self.question_test[idx], self.question_test_mask[idx], self.Y_test[idx]
                predictions_test = self.classify([x], [word_mask], q, mask_question)

#                 if e % 5 == 0:
#                     if predictions_test != y:
#                         print(" wrong, this is question: ", self.idx2sentence(q))
#                         print(" Wrong, this is sentence: ", self.idx2sentence(x[0]))
#                         print(" sentence mask: ", sentence_mask)
#                         print(" word mask: ", word_mask)
#                                                 
#                         print(" Wrong! this is predction: ", self.idx2word[int(predictions_test)], " and this y: ", self.idx2word[int(y)])
#                     else:
#                         print("This is predction: ", self.idx2word[int(predictions_test)], " and this y: ", self.idx2word[int(y)])
#                     
                if predictions_test == y:
                    correct += 1
                total_tests += 1

            print("epoch , " , e, " training ll: ", ll, " ll improvement: ", last_ll - ll, " ratio correct: ", correct / total_tests)
#
            if ll < min_ll_seen:
                min_ll_seen = ll
            if correct / total_tests > best_correct_seen:
                best_correct_seen = correct / total_tests
                
            print(" best ll so far: ", min_ll_seen, " and best ratio: ", best_correct_seen)
            
            last_ll = ll
  
            if last_ll < ll:
                pass
#                 #lr = 0.95 * lr            
#             else:
#                 lr *= 1
#             
#             if e % 25 == 0:
#                 lr /= 2
                        


    def preprocess_babi_set_for_dmn(self):

        filename_train = 'qa1_single-supporting-fact_train.txt'
        filename_test = 'qa1_single-supporting-fact_test.txt'

        filename_output_train = 'babi_train1.txt'
        filename_output_test = 'babi_test1.txt'

        self._write_file(filename_train, filename_output_train)
        self._write_file(filename_test, filename_output_test)

    def _write_file(self, filename_input, filename_output):
        max_babi_article_len = 15
        version = "simple"

        with open(filename_input, encoding='utf-8') as a, open(filename_output, 'w+') as b:
            cur_idx, cur_qa_set, cur_article = 0, [], []
            for line in a:
                if "?" not in line:
                    cur_article.append((''.join([i for i in line if not i.isdigit()]))[1:])
                else:
                    answer_question = re.split(r'\t+', line.strip())
                    cur_name = answer_question[0].split()[3][:-1]

                    if version == "simple":
                        written_sents = []
                        for sent in reversed(cur_article):
                            if cur_name in sent:
                                written_sents.append(sent)
                                break
                        for sent in reversed(cur_article):
                            if cur_name not in sent:
                                written_sents.append(sent)
                                break
                        flipped = 0
                        if len(written_sents) > 1:
                                                        
                            if random.random() < 0.5:
                                flipped = 1
                                tmp = written_sents[0]
                                written_sents[0] = written_sents[1]
                                written_sents[1] = tmp
                        for w in written_sents:
                            b.write(w)
                                                    
                        b.write("$ " + str(flipped) +"\n")
                    else:
                        for sent in cur_article:
                            b.write(sent)

                    b.write("@ " + answer_question[0][2:-2] +"\n")
                    b.write("? " + answer_question[1] + "\n")
                cur_idx += 1
                if cur_idx == max_babi_article_len:
                    cur_idx, cur_qa_set, cur_article = 0, [], []


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

                    Question_test.append((cur_question))
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


    def initialize_dmn_params(self, nh, num_hidden_units_words, num_hidden_units_facts, num_hidden_units_episodes, num_hidden_units_questions, dimension_word_embeddings, dimension_fact_embeddings, max_fact_seqlen, max_number_of_episodes_read, number_word_classes, total_number_of_sentences_per_episode):

        # Initializers
        self.emb = theano.shared(name='embeddings_prob', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (number_word_classes, dimension_word_embeddings)).astype(theano.config.floatX))
        #self.h0_facts_reading_1 = theano.shared(name='h0_facts', value=np.zeros(nh, dtype=theano.config.floatX))
        #self.h0_facts_reading_2 = theano.shared(name='h0_facts', value=np.zeros(nh, dtype=theano.config.floatX))
        #self.h0_facts = [self.h0_facts_reading_1]
        #self.h0_episodes = theano.shared(name='h0_episodes', value=np.zeros(num_hidden_units_episodes, dtype=theano.config.floatX))

        #self.emb_helper = theano.shared(name='embeddings_prob_helper', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (number_word_classes, dimension_word_embeddings)).astype(theano.config.floatX))
        #self.h0_facts_reading_1_helper = theano.shared(name='h0_facts_helper', value=np.zeros(nh, dtype=theano.config.floatX))
        #self.h0_facts_reading_2_helper = theano.shared(name='h0_facts_helper', value=np.zeros(nh, dtype=theano.config.floatX))
        #self.h0_facts_helper = [self.h0_facts_reading_1]
        #self.h0_episodes_helper = theano.shared(name='h0_episodes_helper', value=np.zeros(num_hidden_units_episodes, dtype=theano.config.floatX))
        #self.initialization_randomization_helper = 0

        # GRU Word Parameters
        self.W_word_reset_gate_h = theano.shared(name='W_word_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_reset_gate_x = theano.shared(name='W_word_reset_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_update_gate_h = theano.shared(name='W_word_update_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_update_gate_x = theano.shared(name='W_word_update_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_hidden_gate_h = theano.shared(name='W_word_hidden_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_hidden_gate_x = theano.shared(name='W_word_hidden_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_words)).astype(theano.config.floatX))

        self.W_word_to_fact_vector = theano.shared(name='W_word_to_fact_vector', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.b_word_to_fact_vector = theano.shared(name='b_word_to_fact_vector', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, num_hidden_units_episodes).astype(theano.config.floatX))
        
        dimension_fact_embeddings = 8
        
        # GRU Episode Parameters
        self.W_episode_reset_gate_h = theano.shared(name='W_episode_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_reset_gate_x = theano.shared(name='W_episode_reset_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, dimension_fact_embeddings)).astype(theano.config.floatX))
        self.W_episode_update_gate_h = theano.shared(name='W_episode_update_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_update_gate_x = theano.shared(name='W_episode_update_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, dimension_fact_embeddings)).astype(theano.config.floatX))
        self.W_episode_hidden_gate_h = theano.shared(name='W_episode_hidden_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_hidden_gate_x = theano.shared(name='W_episode_hidden_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, dimension_fact_embeddings)).astype(theano.config.floatX))

        # W_episode to word is (21 x 8)
        self.W_episode_to_word = theano.shared(name='W_fact_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (number_word_classes, num_hidden_units_facts)).astype(theano.config.floatX))
        self.b_episode_to_word = theano.shared(name='W_fact_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, number_word_classes).astype(theano.config.floatX))

        # DMN Gate Parameters
        num_rows_z_dmn = 2
        inner_dmn_dimension = 8
        
        # W_dmn_2:  size (8, 2).  W_dmn_1 size (8,1), b_dmn_1 = 1, b_dmn-2 = 1 
            
        self.W_dmn_1 = theano.shared(name='W_dmn_1', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts , 1)).astype(theano.config.floatX))
        self.W_dmn_2 = theano.shared(name='W_dmn_2', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, total_number_of_sentences_per_episode)).astype(theano.config.floatX))

        self.b_dmn_1 = theano.shared(name='b_dmn_1', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (1)).astype(theano.config.floatX))
        self.b_dmn_2 = theano.shared(name='b_dmn_2', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (1)).astype(theano.config.floatX))
        #self.W_dmn_b = theano.shared(name='W_dmn_2', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, num_hidden_units_questions)).astype(theano.config.floatX))

        self.params = [self.W_word_reset_gate_h, self.W_word_reset_gate_x, self.W_word_update_gate_h, self.W_word_update_gate_x, self.W_word_hidden_gate_h, self.W_word_hidden_gate_x,
                       self.W_word_to_fact_vector, self.b_word_to_fact_vector, self.W_episode_reset_gate_h, self.W_episode_reset_gate_x, self.W_episode_update_gate_h, self.W_episode_update_gate_x,
                       self.W_episode_hidden_gate_h, self.W_episode_hidden_gate_x, self.W_episode_to_word, self.b_episode_to_word, self.W_dmn_1, self.W_dmn_2, self.b_dmn_1, self.b_dmn_2]

        # Question GRU Parameters
        self.W_question_reset_gate_h = theano.shared(name='W_question_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_reset_gate_x = theano.shared(name='W_question_reset_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_update_gate_h = theano.shared(name='W_question_update_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_update_gate_x = theano.shared(name='W_question_update_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_hidden_gate_h = theano.shared(name='W_question_hidden_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_hidden_gate_x = theano.shared(name='W_question_hidden_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_questions)).astype(theano.config.floatX))

        self.W_question_to_vector = theano.shared(name='W_question_to_vector', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.b_question_to_vector = theano.shared(name='b_question_to_vector', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, num_hidden_units_episodes).astype(theano.config.floatX))

        self.h0_questions = theano.shared(name='h0_questions', value=np.zeros(num_hidden_units_questions, dtype=theano.config.floatX))

        self.params.extend((self.W_question_reset_gate_h, self.W_question_reset_gate_x, self.W_question_update_gate_h, self.W_question_update_gate_x, self.W_question_hidden_gate_h, self.W_question_hidden_gate_x,
                            self.W_question_to_vector, self.b_question_to_vector, self.h0_questions))
        
        self.fact_params = [self.W_question_reset_gate_x, self.W_question_reset_gate_h, self.W_question_update_gate_h, self.W_question_update_gate_x, self.W_question_hidden_gate_h, self.W_question_hidden_gate_x, 
                            self.W_question_to_vector, self.b_question_to_vector, self.h0_questions, self.emb, self.W_dmn_1, self.W_dmn_2, self.b_dmn_1, self.b_dmn_2, self.W_word_to_fact_vector, self.b_word_to_fact_vector,                                    
                            self.W_word_reset_gate_h, self.W_word_reset_gate_x, self.W_word_update_gate_h, self.W_word_update_gate_x, self.W_word_hidden_gate_h, self.W_word_hidden_gate_x]


