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

class DMN_No_Scan(object):

    # We take as input a string of "facts"
    def __init__(self, num_fact_hidden_units, number_word_classes, dimension_fact_embeddings, num_episode_hidden_units, max_number_of_facts_read):
        print(" Starting dmn no scan... ")
        #self.preprocess_babi_set_for_dmn()

        print(" Starting dmn no scidxsan... ")
        
        self.X_train, self.mask_sentences_train, self.mask_articles_train, self.question_train, self.question_train_mask, self.Y_train, self.X_test, self.mask_sentences_test, self.mask_articles_test, self.question_test, self.question_test_mask, self.Y_test, word2idx, self.idx2word, dimension_fact_embeddings, max_queslen, max_sentlen, max_article_len = self.process_data("embeddings")
               
        print(" Building model... ")
        number_word_classes = max(self.idx2word.keys(), key=int) + 1
        max_fact_seqlen = max_article_len
        dimension_word_embeddings = 10
        max_number_of_episodes_read = 1
        self.initialization_randomization = 1

        nh = 7 # Dimension of the hidden layer
        num_hidden_units = nh
        num_hidden_units_facts = num_hidden_units
        num_hidden_units_episodes = num_hidden_units_facts
        num_hidden_units_questions = num_hidden_units_episodes
        num_hidden_units_words = num_hidden_units_questions

        self.emb = theano.shared(name='embeddings_prob', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (number_word_classes, dimension_word_embeddings)).astype(theano.config.floatX))

        self.h0_facts_reading_1 = theano.shared(name='h0_facts', value=np.zeros(nh, dtype=theano.config.floatX))
        self.h0_facts_reading_2 = theano.shared(name='h0_facts', value=np.zeros(nh, dtype=theano.config.floatX))

        # self.h0_facts = theano.shared(name='h0_facts', value=np.zeros(nh, dtype=theano.config.floatX))
        self.h0_facts = [self.h0_facts_reading_1, self.h0_facts_reading_2]
        self.h0_episodes = theano.shared(name='h0_episodes', value=np.zeros(num_hidden_units_episodes, dtype=theano.config.floatX))
        #self.h0 = theano.shared(name='h0', value=np.zeros(num_hidden_units_episodes, dtype=theano.config.floatX))

        word_mask = T.lmatrix("word_mask")
        sentence_mask = T.lvector("sentence_mask")

        word_idxs = T.lmatrix("fact_indices") # as many columns as words in the context window and as many lines as words in the sentence
        #x = self.emb[word_idxs].reshape((word_idxs.shape[0], de*cs)) # x basically represents the embeddings of the words IN the current sentence.  So it is shape
        y_sentence = T.lscalar('y_sentence')

        #self.W_fact_to_hidden = theano.shared(name='W_fact_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_fact_embeddings, num_hidden_units_facts)).astype(theano.config.floatX))
        #self.W_hidden_to_hidden = theano.shared(name='W_fact_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, num_hidden_units_facts)).astype(theano.config.floatX))

        # GRU Word Parameters
        self.W_word_reset_gate_h = theano.shared(name='W_word_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_reset_gate_x = theano.shared(name='W_word_reset_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_update_gate_h = theano.shared(name='W_word_update_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_update_gate_x = theano.shared(name='W_word_update_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_hidden_gate_h = theano.shared(name='W_word_hidden_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_hidden_gate_x = theano.shared(name='W_word_hidden_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_words)).astype(theano.config.floatX))

        self.W_word_to_fact_vector = theano.shared(name='W_word_to_fact_vector', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.b_word_to_fact_vector = theano.shared(name='b_word_to_fact_vector', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, num_hidden_units_episodes).astype(theano.config.floatX))

        #self.h0_words = theano.shared(name='h0_episodes', value=np.zeros(num_hidden_units_words, dtype=theano.config.floatX))
        self.h0_words = theano.shared(name='h0_episodes', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (max_number_of_episodes_read, max_fact_seqlen, num_hidden_units_words)).astype(theano.config.floatX))

        # GRU Fact Parameters
        self.W_fact_reset_gate_h = theano.shared(name='W_fact_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, num_hidden_units_facts)).astype(theano.config.floatX))
        self.W_fact_reset_gate_x = theano.shared(name='W_fact_reset_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, dimension_fact_embeddings)).astype(theano.config.floatX))
        self.W_fact_update_gate_h = theano.shared(name='W_fact_update_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, num_hidden_units_facts)).astype(theano.config.floatX))
        self.W_fact_update_gate_x = theano.shared(name='W_fact_update_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, dimension_fact_embeddings)).astype(theano.config.floatX))
        self.W_fact_hidden_gate_h = theano.shared(name='W_fact_hidden_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, num_hidden_units_facts)).astype(theano.config.floatX))
        self.W_fact_hidden_gate_x = theano.shared(name='W_fact_hidden_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, dimension_fact_embeddings)).astype(theano.config.floatX))

        self.W_fact_to_episode = theano.shared(name='W_fact_to_episode', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_facts)).astype(theano.config.floatX))
        self.b_fact_to_episode = theano.shared(name='b_fact_to_episode', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, num_hidden_units_episodes).astype(theano.config.floatX))

        # GRU Episode Parameters
        self.W_episode_reset_gate_h = theano.shared(name='W_episode_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_reset_gate_x = theano.shared(name='W_episode_reset_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, dimension_fact_embeddings)).astype(theano.config.floatX))
        self.W_episode_update_gate_h = theano.shared(name='W_episode_update_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_update_gate_x = theano.shared(name='W_episode_update_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, dimension_fact_embeddings)).astype(theano.config.floatX))
        self.W_episode_hidden_gate_h = theano.shared(name='W_episode_hidden_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_hidden_gate_x = theano.shared(name='W_episode_hidden_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, dimension_fact_embeddings)).astype(theano.config.floatX))

        self.W_out = theano.shared(name='W_fact_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (number_word_classes, num_hidden_units_facts)).astype(theano.config.floatX))
        self.b_out = theano.shared(name='W_fact_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, number_word_classes).astype(theano.config.floatX))

        # DMN Gate Parameters
        num_rows_z_dmn = 2
        inner_dmn_dimension = 8
        self.W_dmn_1 = theano.shared(name='W_dmn_1', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (inner_dmn_dimension , num_rows_z_dmn)).astype(theano.config.floatX))
        self.W_dmn_2 = theano.shared(name='W_dmn_2', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, inner_dmn_dimension)).astype(theano.config.floatX))

        self.b_dmn_1 = theano.shared(name='b_dmn_1', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, num_hidden_units_facts).astype(theano.config.floatX))
        self.b_dmn_2 = theano.shared(name='b_dmn_2', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, num_hidden_units_facts).astype(theano.config.floatX))
        self.W_dmn_b = theano.shared(name='W_dmn_2', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, num_hidden_units_questions)).astype(theano.config.floatX))

        self.params = [self.emb, self.W_out, self.b_out, self.W_fact_reset_gate_h, self.W_fact_reset_gate_x, self.W_fact_update_gate_h,
                       self.W_fact_update_gate_x, self.W_fact_hidden_gate_h, self.W_fact_hidden_gate_x, self.W_fact_to_episode, self.b_fact_to_episode,
                       self.W_episode_reset_gate_h, self.W_episode_reset_gate_x, self.W_episode_update_gate_h, self.W_episode_update_gate_x, self.W_episode_hidden_gate_h, self.W_episode_hidden_gate_x,
                       self.W_dmn_1, self.W_dmn_2, self.b_dmn_1, self.b_dmn_2, self.W_word_reset_gate_h, self.W_word_reset_gate_x, self.W_word_update_gate_h, self.W_word_update_gate_x,
                       self.W_word_hidden_gate_h, self.W_word_hidden_gate_x, self.W_word_to_fact_vector, self.b_word_to_fact_vector, self.h0_words]

        question_encoding = self.GRU_question(dimension_fact_embeddings, num_hidden_units_questions, num_hidden_units_episodes, max_queslen, dimension_word_embeddings)

        def slice_w(x, n):
            return x[n*num_hidden_units_facts:(n+1)*num_hidden_units_facts]

        def word_step(x_cur_word, h_prev, w_mask):

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

        def fact_step(idx, jdx, h_prev, f_mask, sentence_mask_local):

            state_word_step = self.h0_words[idx][jdx]
            for kdx in range(max_sentlen):
                state_word_step = word_step(self.emb[word_idxs[jdx][kdx]], state_word_step, sentence_mask_local[kdx])  # The error would be that self.emb is producing 1,29 and not 1,7

            x_cur = T.tanh(T.dot(self.W_word_to_fact_vector, state_word_step) + self.b_word_to_fact_vector)

            W_in_stacked = T.concatenate([self.W_fact_reset_gate_x, self.W_fact_update_gate_x, self.W_fact_hidden_gate_x], axis=1)
            W_hid_stacked = T.concatenate([self.W_fact_reset_gate_h, self.W_fact_update_gate_h, self.W_fact_hidden_gate_h], axis=1)

            input_n = T.dot(x_cur, W_in_stacked)
            hid_input = T.dot(h_prev, W_hid_stacked)

            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = T.tanh(resetgate)
            updategate = T.tanh(updategate)

            hidden_update = slice_w(input_n, 2) + resetgate * slice_w(hid_input, 2)
            hidden_update = T.tanh(hidden_update)
            h_cur = (1 - updategate) * hidden_update + updategate * hidden_update

            z_dmn = T.concatenate(([question_encoding], [x_cur]), axis=0)

            G_dmn = T.nnet.sigmoid(T.dot(self.W_dmn_2, T.tanh(T.dot(self.W_dmn_1, z_dmn)) + self.b_dmn_1) + self.b_dmn_2)
            h_cur = T.dot(G_dmn, h_cur) + T.dot((1 - G_dmn), h_prev)

            h_cur = f_mask * h_cur + (1 - f_mask) * h_prev
            # h_cur = T.tanh(T.dot(self.W_fact_to_hidden, x_cur) + T.dot(self.W_hidden_to_hidden, h_prev))
            return h_cur

        def episode_step(idx, h_prev, h0_fact):

            state_fact_step = h0_fact
            for jdx in range(max_fact_seqlen):
                state_fact_step = fact_step(idx, jdx, state_fact_step, sentence_mask[jdx], word_mask[jdx])
                #state_fact_step = fact_step(x[jdx], state_fact_step, sentence_mask[jdx], word_mask[jdx])

            x_cur = T.tanh(T.dot(self.W_fact_to_episode, state_fact_step) + self.b_fact_to_episode)

            W_in_stacked = T.concatenate([self.W_episode_reset_gate_x, self.W_episode_update_gate_x, self.W_episode_hidden_gate_x], axis=1)
            W_hid_stacked = T.concatenate([self.W_episode_reset_gate_h, self.W_episode_update_gate_h, self.W_episode_hidden_gate_h], axis=1)

            input_n = T.dot(x_cur, W_in_stacked)
            hid_input = T.dot(h_prev, W_hid_stacked)

            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = T.tanh(resetgate)
            updategate = T.tanh(updategate)

            hidden_update = slice_w(input_n, 2) + resetgate * slice_w(hid_input, 2)
            hidden_update = T.tanh(hidden_update)
            h_cur = (1 - updategate) * hidden_update + updategate * hidden_update

            # h_cur = T.tanh(T.dot(self.W_fact_to_hidden, x_cur) + T.dot(self.W_hidden_to_hidden, h_prev))
            return h_cur

        #state_episode_step = self.h0_episodes  # Could give rise to problem if dimension is not correct
        state_episode_step = question_encoding
        # Reading over the facts
        for idx in range(max_number_of_episodes_read):
            state_episode_step = episode_step(idx, state_episode_step, self.h0_facts[idx])

        output = T.nnet.softmax(T.dot(self.W_out, state_episode_step) + self.b_out)
        p_y_given_x_sentence = output[0, :]

        # err = (state - y_sentence) ** 2
        # updates = theano.OrderedUpdates()
        y_pred = T.argmax(p_y_given_x_sentence, axis=0)
        lr = T.scalar('lr')
        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)  # Returns gradients of the nll w.r.t the params
        sentence_updates = OrderedDict((p, p - lr*g) for p, g in zip(self.params, sentence_gradients))  # computes the update for each of the params.

        print("Compiling fcns...")
        self.classify = theano.function(inputs=[word_idxs, sentence_mask, word_mask, self.question_idxs, self.question_mask], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[word_idxs, sentence_mask, word_mask, self.question_idxs, self.question_mask, y_sentence, lr], outputs=sentence_nll, updates=sentence_updates)
        print("Done compiling!")


    def GRU_question(self, dimension_fact_embedding, num_hidden_units_questions, num_hidden_units_episodes, max_question_len, dimension_word_embeddings):

        # GRU Parameters
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
        for e in range(max_epochs):
            
            shuffled_idxs = [i for i in range(len(self.X_train))]           
            shuffle(shuffled_idxs)
            ll = 0
            num_train_correct, tot_num_train = 0, 0
            
            for idx in shuffled_idxs:
                x, word_mask, sentence_mask, q, mask_question, y = self.X_train[idx], self.mask_sentences_train[idx], self.mask_articles_train[idx], self.question_train[idx], self.question_train_mask[idx], self.Y_train[idx]
                ll += self.sentence_train(x, sentence_mask, word_mask, q, mask_question, y, lr)
            
            shuffle(shuffled_idxs)
            for idx in shuffled_idxs:
                x, word_mask, sentence_mask, q, mask_question, y = self.X_train[idx], self.mask_sentences_train[idx], self.mask_articles_train[idx], self.question_train[idx], self.question_train_mask[idx], self.Y_train[idx]
                predictions_test = self.classify(x, sentence_mask, word_mask, q, mask_question)
                if predictions_test == y:
                    num_train_correct += 1
                tot_num_train += 1

#                 if e == 3 and 100 <= idx <= 104:
#                     print( " x train: ", self.idx2sentence(x[0]))
#                     print(" word mask: ", word_mask)
#                     print(" sentnece mask: ", sentence_mask)
#                     print(" q: ", self.idx2sentence(q))

            print(" ratio training data predicted correctly: ", num_train_correct / tot_num_train)

            correct = 0
            total_tests = 0
            for idx in range(len(self.X_test)):
                x, word_mask, sentence_mask, q, mask_question, y = self.X_test[idx], self.mask_sentences_test[idx], self.mask_articles_test[idx], self.question_test[idx], self.question_test_mask[idx], self.Y_test[idx]
                predictions_test = self.classify(x, sentence_mask, word_mask, q, mask_question)

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
            if last_ll < ll:
                lr = 0.98 * lr            
            else:
                lr *= 1
            
#             if e % 25 == 0:
#                 lr /= 2
                        
            last_ll = ll


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
                        if len(written_sents) > 1:
                            if random.random() < 0.5:
                                tmp = written_sents[0]
                                written_sents[0] = written_sents[1]
                                written_sents[1] = tmp
                        for w in written_sents:
                            b.write(w)                            
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
                if "?" not in line and "@" not in line:
                    cur_sentence = []
                    for w in tokenizer.tokenize(line):
                        cur_sentence.append(w.strip().lower())
                    cur_article.append(cur_sentence)
                    if len(cur_sentence) > max_sentence_len:
                        max_sentence_len = len(cur_sentence)
                else:
                    question_phrase = line[2:].split()
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
                if "?" not in line and "@" not in line:
                    cur_sentence = []
                    for w in tokenizer.tokenize(line):
                        cur_sentence.append(w.strip().lower())
                    cur_article.append(cur_sentence)
                    if len(cur_sentence) > max_sentence_len:
                        max_sentence_len = len(cur_sentence)
                else:
                    question_phrase = line[2:].split()
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

        for article in X_train:
            cur_mask_article = np.zeros(max_article_len, dtype='int32')
            cur_mask_article[0:len(article)] = 1
            mask_articles_train.append(cur_mask_article)

            set_of_sentence_masks = np.zeros((max_article_len, max_sentence_len),dtype='int32')
            for idx, sentence in enumerate(article):
                set_of_sentence_masks[idx, 0:len(sentence)] = 1
            mask_sentences_train.append(set_of_sentence_masks)

        for article in X_test:
            cur_mask_article = np.zeros(max_article_len, dtype='int32')
            cur_mask_article[0:len(article)] = 1
            mask_articles_test.append(cur_mask_article)

            set_of_sentence_masks = np.zeros((max_article_len, max_sentence_len),dtype='int32')
            for idx, sentence in enumerate(article):
                set_of_sentence_masks[idx, 0:len(sentence)] = 1
            mask_sentences_test.append(set_of_sentence_masks)

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

        # # These are to add zeros where we don't currently have elements
        # new_x_train_vec = []
        # for f in X_train_vec:
        #     for added_el in range(len(f), max_article_len):
        #         f = np.concatenate((f, [[0]]), axis=0)
        #     new_x_train_vec.append(f)
        # X_train_vec = new_x_train_vec
        #
        # new_x_test_vec = []
        # for f in X_test_vec:
        #     for added_el in range(len(f), max_article_len):
        #         f = np.concatenate((f, [[0]]), axis=0)
        #     new_x_test_vec.append(f)
        # X_test_vec = new_x_test_vec

        assert(len(X_test_vec) == len(Y_test_vec))

        return X_train_vec, mask_sentences_train, mask_articles_train, Question_train_vec, Question_train_mask, Y_train_vec, X_test_vec, mask_sentences_test, mask_articles_test, Question_test_vec, Question_test_mask, Y_test_vec, word2idx, idx2word, len(word2idx), max_queslen, max_sentence_len, max_article_len



