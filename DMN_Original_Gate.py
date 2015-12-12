__author__ = 'Dan'

import numpy as np
import re
import theano
import theano.tensor as T
import theano.typed_list
from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
import random
from random import shuffle

class DMN_Original_Gate(object):

    # We take as input a string of "facts"
    def __init__(self):

        self.n_batches = 20
        print("DMN Batched Starting dmn no scan... ")
        self.preprocess_babi_set_for_dmn()

        self.X_train, self.mask_sentences_train, self.fact_ordering_train, self.question_train, self.question_train_mask, self.Y_train, self.X_test, self.mask_sentences_test, self.fact_ordering_test, self.question_test, self.question_test_mask, self.Y_test, self.word2idx, self.idx2word, dimension_fact_embeddings, max_queslen, max_sentlen, total_sequence_length = self.process_data("embeddings")


        #self.GRU_x_train, self.GRU_x_test, self.GRU_w_mask_train, self.GRU_w_mask_test = [], [], [], []

        # for x, xm, q, qm in zip(self.X_train, self.mask_sentences_train, self.question_train, self.question_train_mask):
        #     self.GRU_x_train.append(np.concatenate((q, x), axis=1))
        #     self.GRU_w_mask_train.append(np.concatenate((qm, xm), axis=1))
        #
        # for x, xm, q, qm in zip(self.X_train, self.mask_sentences_test, self.question_test, self.question_test_mask):
        #     self.GRU_x_train.append(np.concatenate((q, x), axis=1))
        #     self.GRU_w_mask_train.append(np.concatenate((qm, xm), axis=1))

        print(" Building model... ")
        number_word_classes = max(self.idx2word.keys(), key=int) + 1

        max_fact_seqlen = max_sentlen
        total_number_of_sentences_per_episode = int(total_sequence_length / max_fact_seqlen)
        self.total_number_sentences = total_number_of_sentences_per_episode

        dimension_word_embeddings = 9
        max_number_of_episodes_read = 1
        self.initialization_randomization = .5
        assert(self.initialization_randomization < 1)
        self.num_features_dmn_gate = 3

        nh = 10 # Dimension of the hidden layer
        self.num_hidden_units = nh
        num_hidden_units = nh
        num_hidden_units_facts = num_hidden_units
        num_hidden_units_episodes = num_hidden_units_facts
        num_hidden_units_questions = num_hidden_units_episodes
        num_hidden_units_words = num_hidden_units_questions
        self.num_word_classes = number_word_classes
        total_len_word_seq = total_sequence_length

        self.initialize_dmn_params(nh, num_hidden_units_words, num_hidden_units_facts, num_hidden_units_episodes, num_hidden_units_questions, dimension_word_embeddings, dimension_fact_embeddings, max_fact_seqlen, max_number_of_episodes_read, number_word_classes, total_number_of_sentences_per_episode)

        # INPUT VARIABLES
        word_idxs = T.lmatrix("word_indices") # Dimensions are (num_batches, sequence_word_idxs)
        word_mask = T.TensorType(dtype='int32', broadcastable=(False, False, True))('word_mask')
        y_sentence = T.lmatrix('y_sentence')  # Dimensions are (answer_for_each_batch),
        fact_outputs = T.lmatrix('fact_outputs') # Dimensions are (fact for each batch)

        lr = T.scalar('lr')

        #hid_init = T.dot(np.ones((self.n_batches, 1)), self.h0_gru)

        W_in_stacked_word = T.concatenate([self.W_word_reset_gate_x, self.W_word_update_gate_x, self.W_word_hidden_gate_x], axis=1)  # I think your issue is that this should have # dim word embeddings
        W_hid_stacked_word = T.concatenate([self.W_word_reset_gate_h, self.W_word_update_gate_h, self.W_word_hidden_gate_h], axis=1)

        question_encoding = self.GRU_question(num_hidden_units_questions, max_queslen)

        def slice_w(x, n):
             return x[:, n*num_hidden_units_facts:(n+1)*num_hidden_units_facts]

        def gru_word_layer(x_cur, h_prev, w_mask):

            input_n = T.dot(x_cur, W_in_stacked_word)  # input_n will have dimension (n_batches, W_in_stacked_n_cols)
            hid_input = T.dot(h_prev, W_hid_stacked_word)

            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = T.tanh(resetgate)
            updategate = T.tanh(updategate)
            hidden_update = slice_w(input_n, 2) + resetgate * slice_w(hid_input, 2)
            hidden_update = T.tanh(hidden_update)

            h_cur = (1 - updategate) * h_prev + updategate * hidden_update
            h_cur = w_mask * h_cur + (1 - w_mask) * h_prev

            return h_cur

        list_of_fact_softmaxes = []
        W_in_stacked_episode = T.concatenate([self.W_episode_reset_gate_x, self.W_episode_update_gate_x, self.W_episode_hidden_gate_x], axis=1)  # I think your issue is that this should have # dim brain embeddings
        W_hid_stacked_episode = T.concatenate([self.W_episode_reset_gate_h, self.W_episode_update_gate_h, self.W_episode_hidden_gate_h], axis=1)

        def gru_episode_layer(cur_sentence_idx, h_prev_episode):

            cur_word_state = question_encoding  # could be right, could be wrong.
            for jdx in range(cur_sentence_idx, cur_sentence_idx + max_sentlen):
                x_cur = self.emb[word_idxs[:, jdx]]
                cur_word_state = gru_word_layer(x_cur, cur_word_state, word_mask[:, jdx])

            cur_features_dmn = 3
            assert(self.num_features_dmn_gate == cur_features_dmn)
            z_dmn = T.concatenate(([question_encoding], [cur_word_state], [h_prev_episode]), axis=0).dimshuffle(1, 0, 2)   # (n_batch, n_features=3, n_hidden)

            innermost_product1 = T.tanh(T.dot(z_dmn, self.W_dmn_1))  # The result of this will be (n_batch, n_features, dmn_hidden_units)
            innermost_product2 = T.dot(innermost_product1, self.W_dmn_2)  # The result of this will be (n_batch, n_features, hidden_units_facts)
            self.G_dmn = T.nnet.sigmoid(innermost_product2)  # The gate should be (n_batches, 1)

            #padded_gate = T.shape_padright(self.result_of_gate, 1)
            # this is (n_batch, n_fact_classes, hidden_units) * (n_batch, n_fact_classes (broadcastable)
            #brain_input = T.sum(current_facts * padded_gate, axis=1)

            input_n = T.dot(cur_word_state, W_in_stacked_episode)  #  Line is fine
            hid_input = T.dot(h_prev_episode, W_hid_stacked_episode)  # This is not the error
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = T.tanh(resetgate)
            updategate = T.tanh(updategate)

            hidden_update = slice_w(input_n, 2) + resetgate * slice_w(hid_input, 2)
            hidden_update =  T.tanh(hidden_update)
            h_cur = (1 - updategate) * h_prev_episode + updategate * hidden_update

            padded_gate = T.shape_padright(self.G_dmn, 1)
            h_cur = padded_gate * h_cur + (1 - padded_gate) * h_prev_episode

            return cur_sentence_idx + max_sentlen, h_cur

        cur_episode_state = question_encoding
        cur_sentence_idx = 0
        for idx in range(self.total_number_sentences):
            cur_sentence_idx, cur_episode_state = gru_episode_layer(cur_sentence_idx, cur_episode_state)

        p_y_given_x_sentence = T.nnet.softmax(T.dot(cur_episode_state, self.W_episode_to_word))

        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        sentence_nll = -T.mean(T.min(T.log(p_y_given_x_sentence) * y_sentence, axis=1))
        #fact_nll = -T.mean(T.min(T.log(self.result_of_gate) * fact_outputs, axis=1))

        sentence_gradients = T.grad(sentence_nll, self.params, disconnected_inputs='warn')  # Returns gradients of the nll w.r.t the params

        #fact_gradients = T.grad(fact_nll, self.params, disconnected_inputs='warn')  # Returns gradients of the nll w.r.t the params
        sentence_gradients, norms = self.max_norm_constraints(sentence_gradients, 0.1)

        sentence_updates = OrderedDict((p, p - lr*g) for p, g in zip(self.params, sentence_gradients))  # computes the update for each of the params.
        #fact_updates = OrderedDict((p, p - lr*g) for p, g in zip(self.params, fact_gradients))  # computes the update for each of the params.

        print("Compiling fcns...")
        self.classify = theano.function(inputs=[word_idxs, word_mask, self.question_idxs, self.question_mask], outputs=y_pred, on_unused_input='warn')
        self.sentence_train = theano.function(inputs=[word_idxs, word_mask, self.question_idxs, self.question_mask, y_sentence, lr], outputs=sentence_nll, updates=sentence_updates, on_unused_input='warn')

        # self.fact_train = theano.function(inputs=[word_idxs, word_mask, self.question_idxs, self.question_mask, fact_outputs, y_sentence, lr], outputs=fact_nll, updates=fact_updates, on_unused_input='warn')

        #self.debug_output = theano.function(inputs=[word_idxs, word_mask, self.question_idxs, self.question_mask, y_sentence, lr], outputs=current_facts, on_unused_input='warn')
        print("Done compiling!")


    def max_norm_constraints(self, updates, max_norm, epsilon=1e-3):
        new_grads, norms_out = [], []
        dtype = np.dtype(theano.config.floatX).type
        for g in updates:
            norms = T.sqrt(T.sum(T.sqr(g), axis=0,keepdims=True))

            target_norms = T.clip(norms, 0, dtype(max_norm))
            constrained_output = (g * (target_norms / (dtype(epsilon) + norms)))
            new_grads.append(constrained_output)
            norms_out.append(norms)
        return new_grads, norms_out


    def GRU_question(self, num_hidden_units_questions, max_question_len):

        self.question_idxs = T.lmatrix("question_indices") # dim:  (n_batch, idx_question_word)
        self.question_mask = T.TensorType(dtype='int32', broadcastable=(False, False, True))('question_mask')

        hid_init_question = T.dot(np.ones((self.n_batches, 1)), self.h0_questions)

        def slice_w(x, n):
            return x[:, n*num_hidden_units_questions:(n+1)*num_hidden_units_questions]

        W_in_stacked = T.concatenate([self.W_question_reset_gate_x, self.W_question_update_gate_x, self.W_question_hidden_gate_x], axis=1)
        W_hid_stacked = T.concatenate([self.W_question_reset_gate_h, self.W_question_update_gate_h, self.W_question_hidden_gate_h], axis=1)

        def question_gru_recursion(x_cur, h_prev, q_mask):

            input_n = T.dot(x_cur, W_in_stacked)
            hid_input = T.dot(h_prev, W_hid_stacked)
            resetgate = slice_w(hid_input, 0) + slice_w(input_n, 0)
            updategate = slice_w(hid_input, 1) + slice_w(input_n, 1)
            resetgate = T.tanh(resetgate)
            updategate = T.tanh(updategate)

            hidden_update = slice_w(input_n, 2) + resetgate * slice_w(hid_input, 2)
            hidden_update = T.tanh(hidden_update)
            h_cur = (1 - updategate) * h_prev + updategate * hidden_update
            h_cur = q_mask * h_cur + (1 - q_mask) * h_prev

            return h_cur

        state = hid_init_question
        for idx in range(max_question_len):
            x_cur = self.emb[self.question_idxs[:, idx]]  # This will produce a matrix of size (n_batch, word_dimensions)
            state = question_gru_recursion(x_cur, state, self.question_mask[:, idx])

        return state

    def idx2sentence(self, x):
        cur_sent = ""
        for w in x:
            cur_sent += " " + self.idx2word[int(w)]
        return cur_sent

    def train(self):
        lr = .01
        max_epochs = 20000
        print(" Starting training...")

        for e in range(max_epochs):

            shuffled_idxs = [i for i in range(len(self.X_train))]
            shuffle(shuffled_idxs)
            ll, ll_fact = 0, 0
            num_train_correct, tot_num_train = 0, 0

            x_batch, x_mask_batch, q_batch, q_mask_batch, y_batch, fact_ordering = [], [], [], [], [], []

            total_num_batches = 0
            for idx in shuffled_idxs:

                x_batch.append(self.X_train[idx])
                x_mask_batch.append(self.mask_sentences_train[idx])
                q_batch.append(self.question_train[idx])
                q_mask_batch.append(self.question_train_mask[idx])
                y_batch.append(self.Y_train[idx])
                fact_ordering.append(self.fact_ordering_train[idx])

                if len(x_batch) == self.n_batches:
                    total_num_batches += 1

                    x_mask_batch, q_mask_batch, y_batch2, fact_ordering2 = self._gen_new_batches(x_mask_batch, q_mask_batch, y_batch, fact_ordering)
                    #ll_fact += self.fact_train(x_batch, x_mask_batch, q_batch, q_mask_batch, fact_ordering2, y_batch2, lr)
                    ll_cur = self.sentence_train(x_batch, x_mask_batch, q_batch, q_mask_batch, y_batch2, lr)
                    ll += ll_cur

                    x_batch, x_mask_batch, q_batch, q_mask_batch, y_batch, fact_ordering = [], [], [], [], [], []

            #if e % 30 == 0:
            #    print(" gate res: ", gate_res)
            #print(" overall ll: ", ll, " and ll fact: " , ll_fact)

            if e % 1000 == 0:
                lr /= 2

            test_batch_idx = 0
            shuffle(shuffled_idxs)
            for idx in shuffled_idxs:

                x_batch.append(self.X_train[idx])
                x_mask_batch.append(self.mask_sentences_train[idx])
                q_batch.append(self.question_train[idx])
                q_mask_batch.append(self.question_train_mask[idx])
                y_batch.append(self.Y_train[idx])
                fact_ordering.append(self.fact_ordering_test[idx])

                if len(x_batch) == self.n_batches:

                    test_batch_idx += 1
                    x_mask_batch, q_mask_batch, y_batch2, fact_ordering2 = self._gen_new_batches(x_mask_batch, q_mask_batch, y_batch, fact_ordering)
                    y_pred = self.classify(x_batch, x_mask_batch, q_batch, q_mask_batch)

                    for idx, yp, ya in zip(range(len(y_pred)), y_pred, y_batch):
                        if yp == ya:
                            num_train_correct += 1
                        tot_num_train += 1
                    x_batch, x_mask_batch, q_batch, q_mask_batch, y_batch, fact_ordering = [], [], [], [], [], []

            print(" at epoch, ", e ," avg one ll : ", ll / total_num_batches)
            print(" ratio training data predicted correctly: ", num_train_correct / tot_num_train)


    def _gen_new_batches(self, x_mask_batch, q_mask_batch, y_batch, fact_batch):
        new_masks = []
        for x in x_mask_batch:
            new_batch = []
            for m_val in x:
                new_batch.append([m_val])
            new_masks.append(np.asarray(new_batch))
        x_mask_batch = new_masks

        new_masks = []
        for q in q_mask_batch:
            new_batch = []
            for m_val in q:
                new_batch.append([m_val])
            new_masks.append(np.asarray(new_batch))
        q_mask_batch = new_masks

        if y_batch is None:
            return x_mask_batch, q_mask_batch, None
        else:
            new_y = []
            for y in y_batch:
                cur_vec = []
                for idx in range(self.num_word_classes):
                    cur_vec.append(0)
                cur_vec[y] = 1
                new_y.append(np.asarray(cur_vec))

            if fact_batch is None:
                return x_mask_batch, q_mask_batch, np.asarray(new_y)
            else:
                new_fact = []
                for f in fact_batch:
                    cur_vec = []
                    for idx in range(self.total_number_sentences):
                        cur_vec.append(0)
                    cur_vec[f[0]] = 1
                    new_fact.append(cur_vec)
                return x_mask_batch, q_mask_batch, np.asarray(new_y), np.asarray(new_fact)

    def initialize_dmn_params(self, nh, num_hidden_units_words, num_hidden_units_facts, num_hidden_units_episodes, num_hidden_units_questions, dimension_word_embeddings, dimension_fact_embeddings, max_fact_seqlen, max_number_of_episodes_read, number_word_classes, total_number_of_sentences_per_episode):

        num_hidden_units = nh

        # Initializers
        self.emb = theano.shared(name='embeddings_prob', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (number_word_classes, dimension_word_embeddings)).astype(theano.config.floatX))

        # GRU Word Parameters
        self.W_word_reset_gate_h = theano.shared(name='W_word_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_reset_gate_x = theano.shared(name='W_word_reset_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_update_gate_h = theano.shared(name='W_word_update_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_update_gate_x = theano.shared(name='W_word_update_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_hidden_gate_h = theano.shared(name='W_word_hidden_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_words, num_hidden_units_words)).astype(theano.config.floatX))
        self.W_word_hidden_gate_x = theano.shared(name='W_word_hidden_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_words)).astype(theano.config.floatX))

        #self.W_output_to_answer = theano.shared(name='W_word_to_fact_vector', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, number_word_classes)).astype(theano.config.floatX))
        #self.b_output_to_answer = theano.shared(name='b_word_to_fact_vector', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, num_hidden_units_episodes).astype(theano.config.floatX))

        #self.h0_gru = theano.shared(name='h0_gru', value=np.zeros((1, num_hidden_units), dtype=theano.config.floatX))

        self.params = [self.emb, self.W_word_reset_gate_h, self.W_word_reset_gate_x, self.W_word_hidden_gate_h, self.W_word_hidden_gate_x,
                       self.W_word_update_gate_h, self.W_word_update_gate_x]

        # Question GRU Parameters
        self.W_question_reset_gate_h = theano.shared(name='W_question_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_reset_gate_x = theano.shared(name='W_question_reset_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_update_gate_h = theano.shared(name='W_question_update_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_update_gate_x = theano.shared(name='W_question_update_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_hidden_gate_h = theano.shared(name='W_question_hidden_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_hidden_gate_x = theano.shared(name='W_question_hidden_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (dimension_word_embeddings, num_hidden_units_questions)).astype(theano.config.floatX))
        self.h0_questions = theano.shared(name='h0_questions', value=np.zeros((1, num_hidden_units_questions), dtype=theano.config.floatX))

        self.params.extend((self.W_question_reset_gate_h, self.W_question_reset_gate_x, self.W_question_update_gate_h, self.W_question_update_gate_x, self.W_question_hidden_gate_h, self.W_question_hidden_gate_x,
                        self.h0_questions))

        # GRU Episode Parameters
        self.W_episode_reset_gate_h = theano.shared(name='W_episode_reset_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_reset_gate_x = theano.shared(name='W_episode_reset_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, dimension_fact_embeddings)).astype(theano.config.floatX))
        self.W_episode_update_gate_h = theano.shared(name='W_episode_update_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_update_gate_x = theano.shared(name='W_episode_update_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, dimension_fact_embeddings)).astype(theano.config.floatX))
        self.W_episode_hidden_gate_h = theano.shared(name='W_episode_hidden_gate_h', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_hidden_gate_x = theano.shared(name='W_episode_hidden_gate_x', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, dimension_fact_embeddings)).astype(theano.config.floatX))
        # W_episode to word is (21 x 8)
        self.W_episode_to_word = theano.shared(name='W_episode_to_word', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, number_word_classes)).astype(theano.config.floatX))
        #self.b_episode_to_word = theano.shared(name='b_episode_to_word', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, number_word_classes).astype(theano.config.floatX))

        self.params.extend((self.W_episode_reset_gate_h, self.W_episode_reset_gate_x, self.W_episode_update_gate_h, self.W_episode_update_gate_x, self.W_episode_hidden_gate_h, self.W_episode_hidden_gate_x,
                            self.W_episode_to_word))


        # W_dmn_2:  size (8, 2).  W_dmn_1 size (8,1), b_dmn_1 = 1, b_dmn-2 = 1

        dmn_hidden_units = 10

        # W_dmn_1 should be (num_hidden_units_facts/question)
        # W_dmn_2 should be (num_features)
        self.W_dmn_1 = theano.shared(name='W_dmn_1', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_dmn_2 = theano.shared(name='W_dmn_2', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (self.num_features_dmn_gate)).astype(theano.config.floatX))
        self.b_dmn_1 = theano.shared(name='b_dmn_1', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (1)).astype(theano.config.floatX))
        self.b_dmn_2 = theano.shared(name='b_dmn_2', value=self.initialization_randomization * np.random.uniform(-1.0, 1.0, (1)).astype(theano.config.floatX))

        self.params.extend((self.W_dmn_1, self.W_dmn_2))




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

