__author__ = 'Dan'
import numpy as np
import theano 
import theano.tensor as T
import theano.typed_list
from collections import OrderedDict


class DMNPureTheano(object):

    # We take as input a string of "facts"
    def __init__(self, num_fact_hidden_units, number_word_classes, number_fact_embeddings, dimension_fact_embeddings, num_episode_hidden_units, max_number_of_facts_read):
        self.max_epochs = 500
            
        self.W_fact_embeddings_to_h = theano.shared(name='W_x',
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
        
        self.W_episode_reset_gate_h = theano.shared(name='W_fact_reset_gate_h',
                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                (num_episode_hidden_units, num_episode_hidden_units))
                                .astype(theano.config.floatX))
        
        self.W_episode_reset_gate_x = theano.shared(name='W_fact_reset_gate_x',
                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                (num_fact_hidden_units, num_episode_hidden_units))
                                .astype(theano.config.floatX))
        
        self.W_episode_update_gate_h = theano.shared(name='W_fact_update_gate_h',
                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                (num_episode_hidden_units, num_episode_hidden_units))
                                .astype(theano.config.floatX))
        
        self.W_episode_update_gate_x = theano.shared(name='W_fact_update_gate_x',
                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                (num_fact_hidden_units, num_episode_hidden_units))
                                .astype(theano.config.floatX)) 
  
        self.W_episode_hidden_gate_h = theano.shared(name='W_fact_hidden_gate_h',
                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                (num_episode_hidden_units, num_episode_hidden_units))
                                .astype(theano.config.floatX))
        
        self.W_episode_hidden_gate_x = theano.shared(name='W_fact_hidden_gate_x',
                                value=0.2 * np.random.uniform(-1.0, 1.0,
                                (num_fact_hidden_units, num_episode_hidden_units))
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
                                value=np.zeros((number_fact_embeddings, num_fact_hidden_units),
                                dtype=theano.config.floatX))
        
        self.h0_episodes = theano.shared(name='h0_episodes',
                                value=np.zeros((max_number_of_facts_read, num_fact_hidden_units),
                                dtype=theano.config.floatX))
               
        self.params = [self.W_fact_embeddings_to_h, self.W_fact_to_episodes, 
                       self.b_facts_to_episodes, self.W_episodes_to_answer, self.b_episodes_to_answers, self.h0_facts]   
                           
        
        #recur_ts = T.iscalar('Recurrent_time_step_var')
        recur_ts = theano.shared(0)
                        
        idxs = T.imatrix()
        mask = T.matrix('mask', dtype=theano.config.floatX)        
        x = self.W_fact_embeddings_to_h[idxs].reshape((idxs.shape[0], dimension_fact_embeddings))
        answer = T.ivector("answers")
        
        fact_sequence = theano.shared(value=np.zeros((num_fact_hidden_units, max_number_of_facts_read), dtype=theano.config.floatX), name="fact_sequence",borrow=False)

        def recurrence(m, x_t, h_t):
            reset_gate_fact = T.nnet.sigmoid(T.dot(x_t, self.W_fact_reset_gate_x) + T.dot(h_t, self.W_fact_reset_gate_h))
            update_gate_fact = T.nnet.sigmoid(T.dot(x_t, self.W_fact_update_gate_x) + T.dot(h_t, self.W_fact_update_gate_h))
            
            hidden_update_in = T.nnet.sigmoid(T.dot(x_t, self.W_fact_hidden_gate_x) + reset_gate_fact * T.dot(h_t, self.W_fact_hidden_gate_h))
        
            h_t_plus = (1 - update_gate_fact) * h_t + update_gate_fact * hidden_update_in
            
            h_t_plus = m[:, None] * h_t_plus + (1 - m[:, None]) * h_t
            
            s_t = T.nnet.sigmoid(T.dot(h_t_plus, self.W_fact_to_episodes) + self.b_facts_to_episodes)            
            return [h_t_plus, s_t]
        
        def recurrent_episodes(x_t, h_t_episode):
            
            [h_facts, s_facts], _ = theano.scan(fn=recurrence, sequences=[mask, x], outputs_info=[self.h0_facts, None], 
                                                n_steps=number_fact_embeddings)
                     
            s_t = s_facts[-1]  
        
            reset_gate_episode = T.nnet.sigmoid(T.dot(s_t, self.W_episode_reset_gate_x) + T.dot(h_t_episode, self.W_episode_reset_gate_h))
            update_gate_episode = T.nnet.sigmoid(T.dot(s_t, self.W_episode_update_gate_x) + T.dot(h_t_episode, self.W_episode_update_gate_h))
            
            hidden_update_in_episode = T.nnet.sigmoid(T.dot(s_t, self.W_episode_hidden_gate_x) + reset_gate_episode * T.dot(h_t_episode, self.W_episode_hidden_gate_h))
        
            h_t = (1 - update_gate_episode) * h_t_episode + update_gate_episode * hidden_update_in_episode
                            
                            
                                          
            output_answer = T.nnet.softmax(T.dot(h_t, self.W_episodes_to_answer) + self.b_episodes_to_answers)
             
            return [h_t, output_answer]
                    
        [h_episodes, s_episodes], scan_updates = theano.scan(fn=recurrent_episodes,
                                                             sequences=[fact_sequence],
                                                             outputs_info=[self.h0_episodes, None],
                                                             n_steps=max_number_of_facts_read)
                    
        p_y_given_x_sentence = s_episodes[:, 0, :]  # Why does this have the dimensions that it does?  
        
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)                  
        self.lr = T.scalar('lr')
        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), answer])  # Note:  x shape
        sentence_gradients = T.grad(sentence_nll, self.params, disconnected_inputs="warn")
        sentence_updates = OrderedDict((p, p - self.lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))
        
        self.classify = theano.function(inputs=[idxs, mask], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[idxs, mask, answer, self.lr],outputs=sentence_nll, updates=sentence_updates)
                    
        
    def train(self, training_examples, test_examples):
        lr = 0.001
        
        try:
            
            for edx in range(self.max_epochs):
                
                for train_idxs in training_examples: 
        
                    answer = training_examples['answer'][train_idxs]
                    x = training_examples['x'][train_idxs]
        
                    #cost = f_grad_shared(x, mask, y)
                    #f_update(lrate)
                    
                    self.sentence_train[x, answer, lr]
                    
                for test_idxs in test_examples:
                                       
                    y_pred = self.classify(test_idxs)
        
                    
                    
                    
                    
                    
        except:
            KeyboardInterrupt
        
        
        
        
        
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






