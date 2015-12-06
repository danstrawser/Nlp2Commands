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

class DMNPureTheano(object):

    # We take as input a string of "facts"
    def __init__(self, num_fact_hidden_units, number_word_classes, dimension_fact_embeddings, num_episode_hidden_units, max_number_of_facts_read):
        self.X_train, self.mask_train, self.question_train, self.Y_train, self.X_test, self.mask_test, self.question_test, self.Y_test, word2idx, idx2word, dimension_fact_embeddings = self.process_data()
        number_word_classes = len(self.question_train[0])
        dimension_fact_embeddings = len(self.X_train[0][0])
        max_fact_seqlen = self.mask_train[0].shape[1]
        
        
        # The following are debug lines to not go into the internal recurrence
        max_number_of_facts_read = 3  # DS NOTE:  This is debug line
        #num_fact_hidden_units = dimension_fact_embeddings
        
                
        sys.setrecursionlimit(30000)
        load_from_pickle = 0
        self.max_epochs = 500
                    
        if not load_from_pickle:
                                
            self.W_fact_embeddings_to_h = theano.shared(name='W_fact_embedding_to_h',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (dimension_fact_embeddings, num_fact_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_fact_reset_gate_h = theano.shared(name='W_fact_reset_gate_h',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_fact_hidden_units, num_fact_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_fact_reset_gate_x = theano.shared(name='W_fact_reset_gate_x',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (dimension_fact_embeddings, num_fact_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_fact_update_gate_h = theano.shared(name='W_fact_update_gate_h',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_fact_hidden_units, num_fact_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_fact_update_gate_x = theano.shared(name='W_fact_update_gate_x',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (dimension_fact_embeddings, num_fact_hidden_units))
                                    .astype(theano.config.floatX)) 
            
            self.W_fact_hidden_gate_h = theano.shared(name='W_fact_hidden_gate_h',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_fact_hidden_units, num_fact_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_fact_hidden_gate_x = theano.shared(name='W_fact_hidden_gate_x',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (dimension_fact_embeddings, num_fact_hidden_units))
                                    .astype(theano.config.floatX)) 
            
            self.W_episode_reset_gate_h = theano.shared(name='W_episode_reset_gate_h',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_episode_hidden_units, num_episode_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_episode_reset_gate_x = theano.shared(name='W_episode_reset_gate_x',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_fact_hidden_units, num_episode_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_episode_update_gate_h = theano.shared(name='W_episode_update_gate_h',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_episode_hidden_units, num_episode_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.W_episode_update_gate_x = theano.shared(name='W_episode_update_gate_x',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_fact_hidden_units, num_episode_hidden_units))
                                    .astype(theano.config.floatX)) 
      
            self.W_episode_hidden_gate_h = theano.shared(name='W_episode_hidden_gate_h',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_episode_hidden_units, num_episode_hidden_units))
                                    .astype(theano.config.floatX))
           
           # DS: Note this is Debuged, you shoul dhave this first parameter but for now are using the second 
#             self.W_episode_hidden_gate_x = theano.shared(name='W_episode_hidden_gate_x',
#                                     value=0.2 * np.random.uniform(-1.0, 1.0,
#                                     (num_fact_hidden_units, num_episode_hidden_units))
#                                     .astype(theano.config.floatX)) 
#          
            self.W_episode_hidden_gate_x = theano.shared(name='W_episode_hidden_gate_x',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (dimension_fact_embeddings, num_episode_hidden_units))
                                    .astype(theano.config.floatX)) 
            
            self.W_h_facts = theano.shared(name='W_h_facts',
                                    value=0.2 * np.random.uniform(-1.0, 1.0,
                                    (num_fact_hidden_units, num_fact_hidden_units))
                                    .astype(theano.config.floatX))
           
            self.W_fact_to_episodes = theano.shared(name='W_fact_to_episodes',
                                   value=0.2 * np.random.uniform(-1.0, 1.0,
                                   (num_fact_hidden_units, num_episode_hidden_units))
                                    .astype(theano.config.floatX))
            
            self.b_facts_to_episodes = theano.shared(name='b_facts_to_episodes',
                                   value=np.zeros(num_episode_hidden_units,
                                   dtype=theano.config.floatX))
                  
            self.W_episodes_to_answer = theano.shared(name='w_episode_to_answer',
                                   value=0.2 * np.random.uniform(-1.0, 1.0,
                                   (num_fact_hidden_units, number_word_classes))
                                    .astype(theano.config.floatX))
            
            self.b_episodes_to_answers = theano.shared(name='b_episodes_to_answers',
                                   value=np.zeros(number_word_classes,
                                   dtype=theano.config.floatX))
            
            self.h0_facts = theano.shared(name='h0_facts',
                                    value=np.zeros((max_fact_seqlen, num_fact_hidden_units),
                                    dtype=theano.config.floatX))
            
            self.h0_episodes = theano.shared(name='h0_episodes',
                                    value=np.zeros( num_episode_hidden_units,
                                    dtype=theano.config.floatX))
    
            self.W_dmn_b = theano.shared(name='W_dmn_b_gate',
                                    value=np.zeros((dimension_fact_embeddings, dimension_fact_embeddings),
                                    dtype=theano.config.floatX))
            
    #         self.W_dmn_b = theano.shared(name='W_dmn_b_gate',
    #                                 value=0,dtype=theano.config.floatX)#         
            #self.W_dmn_b = T.scalar('W_dmn_b_gate', dtype=theano.config.floatX)
            #self.W_dmn_b = theano.shared(np.cast['float32'])(0)
            self.W_dmn_b = theano.shared(np.float64(0))
            
            
            size_z_dmn = 9
            self.W_dmn_2 = theano.shared(name='W_dmn_2_gate',
                                    value=np.zeros((max_fact_seqlen, size_z_dmn),
                                    dtype=theano.config.floatX))
            
            self.W_dmn_1 = theano.shared(name='W_dmn_1_gate',
                                    value=np.zeros((max_fact_seqlen, max_fact_seqlen),
                                    dtype=theano.config.floatX))
            
            self.b_dmn_1 = theano.shared(name='b_dmn_1_gate',
                                    value=np.zeros((max_fact_seqlen, 1),
                                    dtype=theano.config.floatX))
            
            self.b_dmn_2 = theano.shared(name='b_dmn_2_gate',
                                    value=np.zeros((max_fact_seqlen, 1),
                                    dtype=theano.config.floatX))
                   
            self.params = [self.W_fact_embeddings_to_h, self.W_fact_reset_gate_h, self.W_fact_reset_gate_x, 
                           self.W_fact_update_gate_h, self.W_fact_update_gate_x, self.W_fact_hidden_gate_h, self.W_fact_hidden_gate_x,
                           self.W_episode_reset_gate_h, self.W_episode_reset_gate_x, self.W_episode_update_gate_h, self.W_episode_update_gate_x,
                           self.W_episode_hidden_gate_h, self.W_episode_hidden_gate_x, self.W_fact_to_episodes, self.b_facts_to_episodes,
                           self.W_episodes_to_answer, self.b_episodes_to_answers, self.h0_facts, self.h0_episodes,
                           self.W_dmn_b, self.W_dmn_1, self.W_dmn_2, self.b_dmn_1, self.b_dmn_2]
                           
            idxs = T.lmatrix()
            mask = T.matrix('mask', dtype=theano.config.floatX)
            questions = T.vector("question", dtype=theano.config.floatX)
            
            #x = self.W_fact_embeddings_to_h[idxs].reshape((idxs.shape[0], dimension_fact_embeddings))
            idxs.tag.test_value = np.random.randint(2, size=(3,dimension_fact_embeddings))
            
            answer = T.ivector("answers")
            
            fact_sequence = theano.shared(value=np.zeros((num_fact_hidden_units, max_number_of_facts_read), dtype=theano.config.floatX), name="fact_sequence",borrow=False)
    
#             def recurrence(m, x_t, h_t, m_t_episode):
#                 reset_gate_fact = T.nnet.sigmoid(T.dot(x_t, self.W_fact_reset_gate_x) + T.dot(h_t, self.W_fact_reset_gate_h))
#                 update_gate_fact = T.nnet.sigmoid(T.dot(x_t, self.W_fact_update_gate_x) + T.dot(h_t, self.W_fact_update_gate_h))
#                 
#                 hidden_update_in = T.nnet.sigmoid(T.dot(x_t, self.W_fact_hidden_gate_x) + reset_gate_fact * T.dot(h_t, self.W_fact_hidden_gate_h))
#             
#                 h_t_plus = (1 - update_gate_fact) * h_t + update_gate_fact * hidden_update_in
#                 h_t_plus = m[:, None] * h_t_plus + (1 - m[:, None]) * h_t
#                 z_dmn = T.concatenate([x_t, m_t_episode, questions, x_t * questions, abs(x_t - questions), abs(x_t - m_t_episode),
#                                             T.cast(x_t * self.W_dmn_b * questions, theano.config.floatX), T.cast(x_t * self.W_dmn_b * m_t_episode, theano.config.floatX)], axis=0)
# 
#                 G_dmn = T.nnet.sigmoid(T.dot(self.W_dmn_2, T.tanh(T.dot(self.W_dmn_1, z_dmn)) + self.b_dmn_1) + self.b_dmn_2)
#                            
#                 h_t_plus = G_dmn * h_t_plus + (1 - G_dmn) * h_t
#                 s_t = T.nnet.sigmoid(T.dot(h_t_plus, self.W_fact_to_episodes) + self.b_facts_to_episodes)            
#                 
#                 return [h_t_plus, s_t]
            
            def recurrent_episodes(x_t, h_t_episode):
                
#                 [h_facts, s_facts], _ = theano.scan(fn=recurrence, sequences=[mask, x], outputs_info=[self.h0_facts, None],
#                                                     non_sequences=h_t_episode[-1], n_steps=number_fact_embeddings)
#                          
#                 s_t = s_facts[-1]  
#                         
#                 reset_gate_episode = T.nnet.sigmoid(T.dot(s_t, self.W_episode_reset_gate_x) + T.dot(h_t_episode, self.W_episode_reset_gate_h))
#                 update_gate_episode = T.nnet.sigmoid(T.dot(s_t, self.W_episode_update_gate_x) + T.dot(h_t_episode, self.W_episode_update_gate_h))
#                 
#                 hidden_update_in_episode = T.nnet.sigmoid(T.dot(s_t, self.W_episode_hidden_gate_x) + reset_gate_episode * T.dot(h_t_episode, self.W_episode_hidden_gate_h))
#             
#                 h_t = (1 - update_gate_episode) * h_t_episode + update_gate_episode * hidden_update_in_episode
#                                               
                
                #h_t_conventional = T.nnet.sigmoid( T.dot( x_t, self.W_episode_hidden_gate_x) + T.dot(h_t_episode, self.W_episode_hidden_gate_h))
                h_t_conventional = h_t_episode
                
                #output_answer = T.nnet.softmax(T.dot(h_t_conventional.T, self.W_episodes_to_answer) + self.b_episodes_to_answers)
                output = T.dot(h_t_conventional.T, self.W_episodes_to_answer) + self.b_episodes_to_answers                
                # output_answer = h_t_conventional
                h_t = h_t_conventional
                return [h_t, output]
                        
            [h_episodes, s_episodes], scan_updates = theano.scan(fn=recurrent_episodes,
                                                                 sequences=idxs, # DS Debug note:  this should be sequences=[fact_sequence]
                                                                 outputs_info=[self.h0_episodes, None],
                                                                 n_steps=max_number_of_facts_read)
                        
    
            
            #p_y_given_x_sentence = s_episodes[:, -1, :]  # Why does this have the dimensions that it does?  
            p_y_given_x_sentence = T.nnet.softmax(s_episodes[-1])
            y_pred = T.argmax(p_y_given_x_sentence, axis=0)           
                   
            self.lr = T.scalar('lr')
            
            sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                                   [T.arange(s_episodes.shape[0]), answer])  # Note:  x shape
            
            
            
            sentence_gradients = T.grad(sentence_nll, self.params, disconnected_inputs="warn")
            
            sentence_updates = OrderedDict((p, p - self.lr*g) for p, g in zip(self.params, sentence_gradients))
                        
                        
            self.classify = theano.function(inputs=[idxs, mask, questions], outputs=y_pred, allow_input_downcast=True, on_unused_input='warn')
            self.sentence_train = theano.function(inputs=[idxs, mask, questions, answer, self.lr],outputs=sentence_nll, updates=sentence_updates, allow_input_downcast=True, on_unused_input='warn')
            
            #f = open("theano_fcns.p", "wb")
            #pickle.dump([self.classify, self.sentence_train], f)
            #f.close()
            
        else:            
            print(" loading pickle...")
            f = open("theano_fcns.p", "rb")
            theano_fcns = pickle.load(f)
            self.classify = theano_fcns[0]
            self.sentence_train = theano_fcns[1]
            
    def train(self):
        # self.X_train, self.mask_train, self.question_train, self.Y_train, self.X_test, self.mask_test, self.question_test, self.Y_test, word2idx, idx2word, dimension_fact_embeddings = self.process_data()
        lr = 0.001
        
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



    def process_data(self):

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
        
        X_train_vec, Question_train_vec, Y_train_vec, X_test_vec, Question_test_vec, Y_test_vec = [], [], [], [], [], []
        
        for s in X_train:
            cur_sentence = []
            for f in s:
                new_vec = np.zeros((len(word2idx)))
                new_vec[word2idx[f]] = 1
                cur_sentence.append(new_vec)
            X_train_vec.append(cur_sentence)
            
        for s in X_train:
            cur_sentence = []
            for f in s:
                new_vec = np.zeros((len(word2idx)))
                new_vec[word2idx[f]] = 1
                cur_sentence.append(new_vec)
            X_test_vec.append(cur_sentence)
        
        for y in Y_train:
            new_vec = np.zeros((len(word2idx)))
            new_vec[word2idx[y]] = 1
            Y_train_vec.append(new_vec)
            
        for y in Y_test:
            new_vec = np.zeros((len(word2idx)))
            new_vec[word2idx[y]] = 1
            Y_test_vec.append(new_vec)
    
        for q in Question_train:
            new_vec = np.zeros((len(word2idx)))
            new_vec[word2idx[q]] = 1
            Question_train_vec.append(new_vec)
       
        for q in Question_test:
            new_vec = np.zeros((len(word2idx)))
            new_vec[word2idx[q]] = 1
            Question_test_vec.append(new_vec)
            
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
            
















