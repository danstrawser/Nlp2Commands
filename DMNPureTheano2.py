from theano.compile.mode import FAST_COMPILE
from numpy import dtype
__author__ = 'Dan'
import numpy as np
import theano 
import theano.tensor as T
import theano.typed_list
from collections import OrderedDict
import pickle
import sys
import os
from theano import config

class DMNPureTheano2(object):




    # We take as input a string of "facts"
    def __init__(self, num_fact_hidden_units, number_word_classes, dimension_fact_embeddings, num_episode_hidden_units, max_number_of_facts_read):

        self.X_train, self.mask_train, self.question_train, self.Y_train, self.X_test, self.mask_test, self.question_test, self.Y_test, word2idx, idx2word, dimension_fact_embeddings = self.process_data("embeddings")


        number_word_classes = max(idx2word.keys(), key=int)
        dimension_fact_embeddings = 5

        nv, de, cs = dimension_fact_embeddings, dimension_fact_embeddings, 1

        nc = number_word_classes
        ne = number_word_classes # Using one hot, the number of embeddings is the same as the dimension of the fact embeddings
        nh = 10 # Dimension of the hidden layer

        # Size:  diemsnion of the embedding x the number of hidden units.  This is because it gets used for x_t * self.wx, where x_t is a word embedding
        self.wx = theano.shared(name='wx', value=0.2 * np.random.uniform(-1.0, 1.0, (de * cs, nh)).astype(theano.config.floatX))

        # Size:  hidden x hidden, for obvious reasons
        self.wh = theano.shared(name='wh', value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))

        # Size:  number of hidden x number of classes, because we output to the number of classes
        self.w = theano.shared(name='w', value=0.2 * np.random.uniform(-1.0, 1.0, (nh, nc)).astype(theano.config.floatX))

        self.bh = theano.shared(name='bh', value=np.zeros(nh, dtype=theano.config.floatX))

        self.b = theano.shared(name='b', value=np.zeros(nc, dtype=theano.config.floatX))

        self.h0 = theano.shared(name='h0', value=np.zeros(nh, dtype=theano.config.floatX))

        self.emb = theano.shared(name='embeddings_prob', value=0.2 * np.random.uniform(-1.0, 1.0, (ne, de)).astype(theano.config.floatX))


        self.params = [self.emb, self.wx, self.wh, self.w, self.bh, self.b, self.h0]

        idxs = T.imatrix() # as many columns as words in the context window and as many lines as words in the sentence
        # x = self.emb[idxs].reshape((idxs.shape[0], de*cs))

        # x basically represents the embeddings of the words IN the current sentence.  So it is shape
        x = self.emb[idxs].reshape((idxs.shape[0], de*cs))

        #y_sentence = T.ivector('y_sentence')  # labels
        y_sentence = T.iscalar('y_sentence')

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[self.h0, None], n_steps=x.shape[0])

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

        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.sentence_train = theano.function(inputs=[idxs, y_sentence, lr], outputs=sentence_nll, updates=sentence_updates)



    def train(self):
        # self.X_train, self.mask_train, self.question_train, self.Y_train, self.X_test, self.mask_test, self.question_test, self.Y_test, word2idx, idx2word, dimension_fact_embeddings = self.process_data()
        lr = 0.1
        max_epochs = 10

        for e in range(max_epochs):

            for idx in range(len(self.X_train)):
                x, q, y = self.X_train[idx], self.question_train[idx], self.Y_train[idx][-1]
                ll = self.sentence_train(x, y, lr)

            for idx in range(len(self.X_test)):
                x, q, y = self.X_test[idx], self.question_test[idx], self.Y_test[idx]

                predictions_test = self.classify(x)

                if e == 9:
                    print(" this is x: ", x)
                    print(" this is prediction : ", predictions_test)
                    print(" this is true: ", y[-1])




        assert(1 == 2)
















        # idxs, mask, questions, answer
        # theano.function(inputs=[idxs, mask, questions, answer, self.lr]
            
        for edx in range(self.max_epochs):
            
            print(" starting training epoch: " , edx)
            
            for facts, answer, mask, question in zip(self.X_train, self.Y_train, self.mask_train, self.question_train): 
                print(" Another training example...")
            
#                     answer = training_examples['answer'][train_idxs]
#                     x = training_examples['x'][train_idxs]
#         
                #cost = f_grad_shared(x, mask, y)
                #f_update(lrate)
                facts = np.asarray(facts)
                print(" facts: ", facts)
                print(" len facts: ", facts.shape)
                print(" mask: ", mask[0])
                print(" question: ", question)
                print(" answer: ", answer)
                print(" lr: ", lr)
                
                print(" size facts: ", len(facts[0]))
                print(" size mask: ", len(mask[0]))
                print(" size question: ", len(question))
                                
                result = self.sentence_train(facts, mask, question, answer, lr)
                print(" Done with training example! And this was the result: ", result)
                
            print(" finished training for epoch: ", edx)
                
            for facts, answer, mask, question in zip(self.X_test, self.Y_test, self.mask_test, self.question_test): 
                #self.classify = theano.function(inputs=[idxs, mask, questions], outputs=y_pred)
                           
                y_pred = self.classify(facts, mask, question)
                
                print(" this is y _pred: ", y_pred)
        
        
    def adadelta(self, lr, tparams, grads, x, mask, y, cost):
        """
        An adaptive learning rate optimizer
    
        Parameters
        ----------
        lr : Theano SharedVariable
            Initial learning rate
        tpramas: Theano SharedVariable
            Model parameters
        grads: Theano variable
            Gradients of cost w.r.t to parameres
        x: Theano variable
            Model inputs
        mask: Theano variable
            Sequence mask
        y: Theano variable
            Targets
        cost: Theano variable
            Objective fucntion to minimize
    
        Notes
        -----
        For more information, see [ADADELTA]_.
    
        .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
           Rate Method*, arXiv:1212.5701.
        """
    
        zipped_grads = [theano.shared(p.get_value() * self.numpy_floatX(0.),
                                      name='%s_grad' % k)
                        for k, p in tparams.iteritems()]
        running_up2 = [theano.shared(p.get_value() * self.numpy_floatX(0.),
                                     name='%s_rup2' % k)
                       for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * self.numpy_floatX(0.),
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.iteritems()]
    
        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]
    
        f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                        name='adadelta_f_grad_shared')
    
        updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]
    
        f_update = theano.function([lr], [], updates=ru2up + param_up,
                                   on_unused_input='ignore',
                                   name='adadelta_f_update')
    
        return f_grad_shared, f_update
            
    def numpy_floatX(self, data):
        return np.asarray(data, dtype=theano.config.floatX)


    def process_data(self, type_of_embedding):

        filename_train = 'data/simple_dmn_theano/data_train.txt'
        filename_test = 'data/simple_dmn_theano/data_test.txt'

        X_train, mask_train, Question_train, Y_train, X_test, mask_test, Question_test, Y_test = [], [], [], [], [], [], [], []

        cur_idx = 0
        word2idx = {}
        idx2word = {}
        
        max_mask_len = 3

        with open(filename_train, encoding='utf-8') as f:
            cur_sentence = []                        
            for idx, line in enumerate(f):
                if "?" not in line:
                    cur_sentence.append(line.strip())                   
                else:
                    Y_train.append(line[2:].strip())
                    Question_train.append("where")                    
                    cur_mask = np.zeros((1, max_mask_len))
                    cur_mask[0:len(cur_sentence)] = 1
                    mask_train.append(cur_mask)
                    X_train.append(cur_sentence) # Since a question marks the end of the current sentence
                    cur_sentence = []

        with open(filename_test, encoding='utf-8') as f:
            cur_sentence = []                        
            for idx, line in enumerate(f):
                if "?" not in line:
                    cur_sentence.append(line.strip())                   
                else:
                    Y_test.append(line[2:].strip())  
                    Question_test.append("where")                  
                    cur_mask = np.zeros((1, max_mask_len))
                    cur_mask[0:len(cur_sentence)] = 1
                    mask_test.append(cur_mask)
                    X_test.append(cur_sentence) # Since a question marks the end of the current sentence
                    cur_sentence = []

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
        
        word2idx["where"] = cur_idx
        idx2word[cur_idx] = "where" 
        cur_idx += 1

        assert(len(X_test) == len(Y_test))

        # if type_of_embedding == "embeddings":
        #     size_of_vocab = cur_idx
        #     return X_train, mask_train, Question_train, Y_train, X_test, mask_test, Question_test, Y_test, size_of_vocab

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
                new_vec = [word2idx[y], word2idx[y], word2idx[y]]
            else:
                new_vec = np.zeros((len(word2idx)))
                new_vec[word2idx[y]] = 1
            Y_train_vec.append(new_vec)
            
        for y in Y_test:
            if type_of_embedding == "embeddings":
                new_vec = [word2idx[y], word2idx[y], word2idx[y]]
            else:
                new_vec = np.zeros((len(word2idx)))
                new_vec[word2idx[y]] = 1
            Y_test_vec.append(new_vec)


        for q in Question_train:
            if type_of_embedding == "embeddings":
                new_vec = word2idx[q]
            else:
                new_vec = np.zeros((len(word2idx)))
                new_vec[word2idx[q]] = 1
            Question_train_vec.append(new_vec)
       
        for q in Question_test:
            if type_of_embedding == "embeddings":
                new_vec = word2idx[q]
            else:
                new_vec = np.zeros((len(word2idx)))
                new_vec[word2idx[q]] = 1
            Question_test_vec.append(new_vec)

        assert(len(X_test_vec) == len(Y_test_vec))
            
        return X_train_vec, mask_train, Question_train_vec, Y_train_vec, X_test_vec, mask_test, Question_test_vec, Y_test_vec, word2idx, idx2word, len(word2idx)


    # Word embeddings size is what we use to represent the words.  For example, if there are 
    def GRULayer(self, dimension_word_embeddings, max_seqlen, num_word_hidden_units, fact_embedding_dimension):

            self.W_word_embeddings_to_h = theano.shared(name='W_word_embeddings',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (dimension_word_embeddings, num_word_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_word_reset_gate_h = theano.shared(name='W_word_reset_gate_h',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_word_hidden_units, num_word_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_word_reset_gate_x = theano.shared(name='W_word_reset_gate_x',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (dimension_word_embeddings, num_word_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_word_update_gate_h = theano.shared(name='W_word_update_gate_h',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_word_hidden_units, num_word_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_word_update_gate_x = theano.shared(name='W_word_update_gate_x',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (dimension_word_embeddings, num_word_hidden_units))
                                    .astype(theano.config.floatX)) 
            
            self.W_word_hidden_gate_h = theano.shared(name='W_word_hidden_gate_h',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_word_hidden_units, num_word_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_word_hidden_gate_x = theano.shared(name='W_word_hidden_gate_x',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (dimension_word_embeddings, num_word_hidden_units))
                                    .astype(theano.config.floatX)) 
            
            self.h0_max_word_seqlen = theano.shared(name='h0_words',
                                    value=np.zeros((max_seqlen, num_word_hidden_units),
                                    dtype=theano.config.floatX))
            
            self.W_hidden_word_to_facts = theano.shared(name='W_hidden_word_to_facts',
                                    value=np.zeros((max_seqlen, fact_embedding_dimension),
                                    dtype=theano.config.floatX))
             
            
            self.params = [self.W_word_embeddings_to_h, self.W_word_reset_gate_h, self.W_word_reset_gate_x, self.W_word_update_gate_h,
                           self.W_word_update_gate_x, self.W_word_hidden_gate_h, self.W_word_update_gate_x, self.h0_max_word_seqlen]

            words = T.imatrix()
            mask = T.matrix('mask', dtype=theano.config.floatX)
            
            def GRURecurrence(x_t, h_t):
                
                reset_gate_word = T.nnet.sigmoid(T.dot(x_t, self.W_word_reset_gate_x) + T.dot(h_t, self.W_word_reset_gate_h))
                update_gate_word = T.nnet.sigmoid(T.dot(x_t, self.W_word_update_gate_x) + T.dot(h_t, self.W_word_update_gate_h))
                
                hidden_update_in_word = T.nnet.sigmoid(T.dot(x_t, self.W_word_hidden_gate_x) + reset_gate_word * T.dot(h_t, self.W_word_hidden_gate_h))
            
                h_t = (1 - update_gate_word) * h_t + update_gate_word * hidden_update_in_word
                                              
                output_answer = T.nnet.softmax(T.dot(h_t, self.W_episodes_to_answer) + self.b_episodes_to_answers)
                 
                return [h_t, output_answer]
            
                        
            [h_episodes, s_episodes], scan_updates = theano.scan(fn=GRURecurrence,
                                                                 sequences=[words],
                                                                 outputs_info=[self.h0_max_word_seqlen, None],
                                                                 n_steps=max_seqlen)