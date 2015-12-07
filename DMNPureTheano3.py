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

class DMNPureTheano3(object):

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

        # Size:  diemsnion of the embedding x the number of hidden units.  This is because it gets used for x_t * self.wx, where x_t is a word embedding
        self.wx = theano.shared(name='wx', value=0.2 * np.random.uniform(-1.0, 1.0, (de * cs, nh)).astype(theano.config.floatX))

        # Size:  number of hidden x number of classes, because we output to the number of classes
        self.w = theano.shared(name='w', value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nc)).astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=np.zeros(nc, dtype=theano.config.floatX))
        self.h0_facts = theano.shared(name='h0_facts', value=np.zeros(nh, dtype=theano.config.floatX))
        self.emb = theano.shared(name='embeddings_prob', value=0.2 * np.random.uniform(-1.0, 1.0, (ne, de)).astype(theano.config.floatX))

        # GRU Parameters
        self.W_fact_reset_gate_h = theano.shared(name='W_fact_reset_gate_h', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, num_hidden_units_facts)).astype(theano.config.floatX))
        self.W_fact_reset_gate_x = theano.shared(name='W_fact_reset_gate_x', value=0.2 * np.random.uniform(-1.0, 1.0, (dimension_fact_embeddings, num_hidden_units_facts)).astype(theano.config.floatX))
        self.W_fact_update_gate_h = theano.shared(name='W_fact_update_gate_h', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, num_hidden_units_facts)).astype(theano.config.floatX))
        self.W_fact_update_gate_x = theano.shared(name='W_fact_update_gate_x', value=0.2 * np.random.uniform(-1.0, 1.0, (dimension_fact_embeddings, num_hidden_units_facts)).astype(theano.config.floatX))
        self.W_fact_hidden_gate_h = theano.shared(name='W_fact_hidden_gate_h', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, num_hidden_units_facts)).astype(theano.config.floatX))
        self.W_fact_hidden_gate_x = theano.shared(name='W_fact_hidden_gate_x', value=0.2 * np.random.uniform(-1.0, 1.0, (dimension_fact_embeddings, num_hidden_units_facts)).astype(theano.config.floatX))

        # TODO:  Make sure these dimensions are correct
        self.W_episode_reset_gate_h = theano.shared(name='W_episode_reset_gate_h', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_reset_gate_x = theano.shared(name='W_episode_reset_gate_x', value=0.2 * np.random.uniform(-1.0, 1.0, (number_word_classes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_update_gate_h = theano.shared(name='W_episode_update_gate_h', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_update_gate_x = theano.shared(name='W_episode_update_gate_x', value=0.2 * np.random.uniform(-1.0, 1.0, (number_word_classes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_hidden_gate_h = theano.shared(name='W_episode_hidden_gate_h', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.W_episode_hidden_gate_x = theano.shared(name='W_episode_hidden_gate_x', value=0.2 * np.random.uniform(-1.0, 1.0, (number_word_classes, num_hidden_units_episodes)).astype(theano.config.floatX))

        self.W_episode_to_answer = theano.shared(name='W_episode_to_answer', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_episodes, number_word_classes)).astype(theano.config.floatX))
        self.b_episode_to_answer = theano.shared(name='b_episode_to_answer', value=0.2 * np.random.uniform(-1.0, 1.0, number_word_classes).astype(theano.config.floatX))
        self.h0_episodes = theano.shared(name='h0_episodes', value=np.zeros(num_hidden_units_episodes, dtype=theano.config.floatX))

        num_rows_z_dmn = 7
        inner_dmn_dimension = 8
        self.W_dmn_1 = theano.shared(name='W_dmn_1', value=0.2 * np.random.uniform(-1.0, 1.0, (inner_dmn_dimension , num_rows_z_dmn)).astype(theano.config.floatX))
        self.W_dmn_2 = theano.shared(name='W_dmn_2', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, inner_dmn_dimension)).astype(theano.config.floatX))

        self.b_dmn_1 = theano.shared(name='b_dmn_1', value=0.2 * np.random.uniform(-1.0, 1.0, num_hidden_units_facts).astype(theano.config.floatX))
        self.b_dmn_2 = theano.shared(name='b_dmn_2', value=0.2 * np.random.uniform(-1.0, 1.0, num_hidden_units_facts).astype(theano.config.floatX))
        self.W_dmn_b = theano.shared(name='W_dmn_2', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_facts, num_hidden_units_questions)).astype(theano.config.floatX))

        self.params = [self.emb, self.w, self.b, self.W_fact_reset_gate_h, self.W_fact_reset_gate_x,
                       self.W_fact_update_gate_x, self.W_fact_update_gate_h, self.W_fact_hidden_gate_x, self.W_fact_hidden_gate_h,
                       self.W_episode_reset_gate_h, self.W_episode_reset_gate_x, self.W_episode_update_gate_h, self.W_episode_update_gate_x,
                       self.W_episode_hidden_gate_h, self.W_episode_hidden_gate_x, self.W_episode_to_answer, self.b_episode_to_answer,
                       self.W_dmn_1, self.W_dmn_2, self.b_dmn_1, self.b_dmn_2]

        fact_mask = T.lvector("fact_mask")
        fact_idxs = T.lmatrix("fact_indices") # as many columns as words in the context window and as many lines as words in the sentence
        x = self.emb[fact_idxs].reshape((fact_idxs.shape[0], de*cs)) # x basically represents the embeddings of the words IN the current sentence.  So it is shape
        y_sentence = T.lscalar('y_sentence')

        question_idxs, question_mask, question = self.GRU_question(dimension_fact_embeddings, num_hidden_units_questions, num_hidden_units_episodes, max_queslen)

        def fact_gru_recursion(mask_t, x_t, h_tm1, m_tm1):

            reset_gate_fact = T.nnet.sigmoid(T.dot(x_t, self.W_fact_reset_gate_x) + T.dot(h_tm1, self.W_fact_reset_gate_h))
            update_gate_fact = T.nnet.sigmoid(T.dot(x_t, self.W_fact_update_gate_x) + T.dot(h_tm1, self.W_fact_update_gate_h))
            hidden_update_in_fact = T.nnet.sigmoid(T.dot(x_t, self.W_fact_hidden_gate_x) + reset_gate_fact * T.dot(h_tm1, self.W_fact_hidden_gate_h))
            h_t = (1 - update_gate_fact) * h_tm1 + update_gate_fact * hidden_update_in_fact

            z_dmn = T.concatenate(([x_t], [m_tm1], [question], [x_t * question], [x_t * m_tm1], [abs(x_t - question)], [abs(x_t - m_tm1)]), axis=0)
                                   # [T.dot(x_t.T, T.dot(self.W_dmn_b, question))], [T.dot(x_t.T, T.dot(self.W_dmn_b, m_tm1))]), axis=0)

            #print(" z dmn: ", z_dmn.ndim)
            # What has shape 8,3:  W_dmn_1

            # This will (hopefully) have size (num_hidden_fact_units, num_hidden_fact_units)
            G_dmn = T.nnet.sigmoid(T.dot(self.W_dmn_2, T.tanh(T.dot(self.W_dmn_1, z_dmn)) + self.b_dmn_1) + self.b_dmn_2)

            #print(" G dim: ", G_dmn.ndim)

            h_t = T.dot(G_dmn, h_t) + T.dot((1 - G_dmn), h_tm1)

            #print(" h_t dim: ", h_t.ndim)

            h_t = mask_t * h_t + (1 - mask_t) * h_tm1

            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        def episode_gru_recursion(h_tm1):

            [h, s_t], _ = theano.scan(fn=fact_gru_recursion, sequences=[fact_mask, x], outputs_info=[self.h0_facts, None], non_sequences=h_tm1, n_steps=x.shape[0])
            x_t = s_t[-1, 0, :]  # Note:  I believe this is (1x11)

            reset_gate_episode = T.nnet.sigmoid(T.dot(x_t, self.W_episode_reset_gate_x) + T.dot(h_tm1, self.W_episode_reset_gate_h))
            update_gate_episode = T.nnet.sigmoid(T.dot(x_t, self.W_episode_update_gate_x) + T.dot(h_tm1, self.W_episode_update_gate_h))
            hidden_update_in_episode = T.nnet.sigmoid(T.dot(x_t, self.W_episode_hidden_gate_x) + reset_gate_episode * T.dot(h_tm1, self.W_episode_hidden_gate_h))
            h_t = (1 - update_gate_episode) * h_tm1 + update_gate_episode * hidden_update_in_episode

            s_t = T.nnet.softmax(T.dot(h_t, self.W_episode_to_answer) + self.b_episode_to_answer)

            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=episode_gru_recursion, sequences=[], outputs_info=[self.h0_episodes, None], n_steps=max_number_of_facts_read)

        # I believe the dimensions of p_y_given_x_sentence are ( time, num_classes)
        p_y_given_x_sentence = s[-1, 0, :]  # I believe the output indexing here is (num_classes, time, number_embeddings)

        # DS Note:  it makes sense that the classes would be the second dimension because we are taking argmax of axis=1
        # Which is what we would want for predicting the most likely class
        y_pred = T.argmax(p_y_given_x_sentence, axis=0)

        lr = T.scalar('lr')

        # My thoughts:
        # For some reason, we need T.arange(x.shape[0]) because p_y_given_x_sentence is larger than the current indices put in so you only want the words given.
        # This might be because you have different sentence sizes and p_y_given_x_sentence is always of the same size (so it's sort of like a mask)

        # y_sentence is an ivector which is (I'm assuming) the tags for the sentence, i.e. [23, 234, 66, 66, 21].
        # and then you get p_y_given_x_sentence for these tag values.  This is because you want to maximize these values only.
        # This is the most cryptic line in the whole thing:

        #sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[T.arange(x.shape[0]), y_sentence])
        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[y_sentence])

        sentence_gradients = T.grad(sentence_nll, self.params)  # Returns gradients of the nll w.r.t the params

        sentence_updates = OrderedDict((p, p - lr*g) for p, g in zip(self.params, sentence_gradients))  # computes the update for each of the params.

        self.classify = theano.function(inputs=[fact_idxs, fact_mask, question_idxs, question_mask], outputs=y_pred)

        self.sentence_train = theano.function(inputs=[fact_idxs, fact_mask, question_idxs, question_mask, y_sentence, lr], outputs=sentence_nll, updates=sentence_updates)


    def train(self):
        # self.X_train, self.mask_train, self.question_train, self.Y_train, self.X_test, self.mask_test, self.question_test, self.Y_test, word2idx, idx2word, dimension_fact_embeddings = self.process_data()
        lr = 1
        max_epochs = 1000

        print(" Starting training...")

        for e in range(max_epochs):

            ll = 0
            for idx in range(len(self.X_train)):
                x, m, q, mask_question, y = self.X_train[idx], self.mask_train[idx], self.question_train[idx], self.question_train_mask[idx], self.Y_train[idx]
                ll += self.sentence_train(x, m, q, mask_question, y, lr)

            correct = 0
            total_tests = 0

            for idx in range(len(self.X_test)):
                x, m, q, mask_question, y = self.X_test[idx], self.mask_test[idx], self.question_test[idx], self.question_test_mask[idx], self.Y_test[idx]

                predictions_test = self.classify(x, m, q, mask_question)

                if idx == 30:
                    print(" prediction test: ", predictions_test)
                    print(" y : ", y)

                if predictions_test == y:
                    correct += 1
                total_tests += 1

            print("epoch , " , e, " training ll: ", ll, " ratio correct: ", correct / total_tests)


    def GRU_question(self, dimension_fact_embedding, num_hidden_units_questions, num_hidden_units_episodes, max_question_len):

        # GRU Parameters
        self.W_question_reset_gate_h = theano.shared(name='W_question_reset_gate_h', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_reset_gate_x = theano.shared(name='W_question_reset_gate_x', value=0.2 * np.random.uniform(-1.0, 1.0, (dimension_fact_embedding, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_update_gate_h = theano.shared(name='W_question_update_gate_h', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_update_gate_x = theano.shared(name='W_question_update_gate_x', value=0.2 * np.random.uniform(-1.0, 1.0, (dimension_fact_embedding, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_hidden_gate_h = theano.shared(name='W_question_hidden_gate_h', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_questions)).astype(theano.config.floatX))
        self.W_question_hidden_gate_x = theano.shared(name='W_question_hidden_gate_x', value=0.2 * np.random.uniform(-1.0, 1.0, (dimension_fact_embedding, num_hidden_units_questions)).astype(theano.config.floatX))

        self.W_question_to_vector = theano.shared(name='W_question_to_vector', value=0.2 * np.random.uniform(-1.0, 1.0, (num_hidden_units_questions, num_hidden_units_episodes)).astype(theano.config.floatX))
        self.b_question_to_vector = theano.shared(name='b_question_to_vector', value=0.2 * np.random.uniform(-1.0, 1.0, num_hidden_units_episodes).astype(theano.config.floatX))

        self.h0_questions = theano.shared(name='h0_questions', value=np.zeros(num_hidden_units_questions, dtype=theano.config.floatX))

        self.params.extend((self.W_question_reset_gate_h, self.W_question_reset_gate_x, self.W_question_update_gate_h, self.W_question_update_gate_x, self.W_question_hidden_gate_h, self.W_question_hidden_gate_x,
                            self.W_question_to_vector, self.b_question_to_vector, self.h0_questions))

        question_idxs = T.lmatrix("question_indices") # as many columns as words in the context window and as many lines as words in the sentence
        question_mask = T.lvector("question_mask")
        q = self.emb[question_idxs].reshape((question_idxs.shape[0], dimension_fact_embedding)) # x basically represents the embeddings of the words IN the current sentence.  So it is shape

        def question_gru_recursion(m_t, x_t, h_tm1):

            reset_gate_question = T.nnet.sigmoid(T.dot(x_t, self.W_question_reset_gate_x) + T.dot(h_tm1, self.W_question_reset_gate_h))
            update_gate_question = T.nnet.sigmoid(T.dot(x_t, self.W_question_update_gate_x) + T.dot(h_tm1, self.W_question_update_gate_h))
            hidden_update_in_question = T.nnet.sigmoid(T.dot(x_t, self.W_question_hidden_gate_x) + reset_gate_question * T.dot(h_tm1, self.W_question_hidden_gate_h))
            h_t = (1 - update_gate_question) * h_tm1 + update_gate_question * hidden_update_in_question

            h_t = m_t * h_t + (1 - m_t) * h_t
            s_t = T.nnet.softmax(T.dot(h_t, self.W_question_to_vector) + self.b_question_to_vector)

            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=question_gru_recursion, sequences=[question_mask, q], outputs_info=[self.h0_questions, None], n_steps=max_question_len)

        return question_idxs, question_mask, s[-1, 0, :]


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