__author__ = 'Dan'


from DMNLayer import DMNLayer
from DMNLayerV2 import DMNLayerV2
from DMNLayerV3 import DMNLayer3
import numpy as np
import theano
import theano.tensor as T
import lasagne
import time

class ConcatenateLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, nonlinearity=None, **kwargs):
        super(ConcatenateLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:2]

    def get_output_for(self, inputs, **kwargs):
        output = T.concatenate(inputs[0], inputs[1])
        if self.nonlinearity is not None:
            output = self.nonlinearity(output)
        return output


class DynamicMemNet(object):

    N_BATCH = 1
    N_HIDDEN = 20
    N_HIDDEN_W2F = 20
    N_HIDDEN_BRAIN = 20
    GRAD_CLIP = 100
    LEARNING_RATE = .001

    N_HIDDEN_H = 20
    N_HIDDEN_M = 20

    def __init__(self, X_train, Q_train, Y_train, mask_train, X_test, Q_test, Y_test, mask_test, input_size, max_seqlen, idx2word, max_queslen):   
                
        self.X_train = X_train
        self.Q_train = Q_train
        self.y_train = Y_train
        self.mask_train = mask_train
        self.X_test = X_test
        self.Q_Test = Q_test
        self.y_test = Y_test
        self.mask_test = mask_test
        self.input_size = input_size
        self.max_seq_len = max_seqlen
        self.idx2word = idx2word
        self.vocab_size = len(idx2word)

        self.max_norm = 40      # Same as the MemNet
        self.init_lr = .001     # Same as the MemNet
        self.lr= self.init_lr   # Same as the MemNet
        self.num_epochs = 100
        
        self.word_embedding_size = input_size
        self.max_number_of_facts = 2
        self.max_question_len = max_queslen

    # Note that you could want to create an embedding for input context
    def build(self, input_var=None):
               
        print(" Initializing Dynamic Mem Net")
        
        input_size, max_seqlen = self.input_size, self.max_seq_len
        vocab_size = self.input_size
        word_embedding_size = self.word_embedding_size
        max_question_len = self.max_question_len
                
        y = T.vector('target_output')
        # Build the network
        l_in = lasagne.layers.InputLayer(shape=(self.N_BATCH, max_seqlen, input_size))
        l_mask = lasagne.layers.InputLayer(shape=(self.N_BATCH, max_seqlen))
        
        word_to_fact_layers = []
        for idx in range(self.max_number_of_facts): # TODO:  This is not number of readings but words in a sentence
            w2f_layer = lasagne.layers.GRULayer(l_in, self.N_HIDDEN_W2F, mask_input=l_mask, grad_clipping=self.GRAD_CLIP, only_return_final=True)     
            word_to_fact_layers.append(lasagne.layers.ReshapeLayer(w2f_layer, (self.N_BATCH, 1, self.N_HIDDEN_W2F)))
        
        l_in_question = lasagne.layers.InputLayer(shape=(self.N_BATCH, max_question_len, word_embedding_size))
        l_in_question_mask = lasagne.layers.InputLayer(shape=(self.N_BATCH, max_question_len))
        
        question_encoding_gru = lasagne.layers.GRULayer(l_in_question, self.N_HIDDEN_W2F, mask_input=l_in_question_mask, grad_clipping=self.GRAD_CLIP, only_return_final=True)
               
        # Note:  need to concatenate word2fact_layer and the question.  
        facts = lasagne.layers.ConcatLayer([fact for fact in word_to_fact_layers], axis=1)
        
        print(" these are the facts: ", lasagne.layers.get_output_shape(facts))
        
        #brain_layer = DMNLayer(facts, question_encoding_gru.get_output_for([l_in_question.input_var]), self.N_HIDDEN_H, self.N_HIDDEN_M, max_seqlen)
        #brain_layer = DMNLayerV2(facts, question_encoding_gru.get_output_for([l_in_question.input_var]), self.N_HIDDEN_H, self.N_HIDDEN_M, max_seqlen)
        brain_layer = DMNLayerV2(facts, question_encoding_gru.get_output_for([l_in_question.input_var]), self.N_HIDDEN_H, self.N_HIDDEN_M, max_seqlen)
                
        #brain_layer = DMNLayer(l_in_question, self.N_HIDDEN_H)
        
        #brain_layer = lasagne.layers.GRULayer(episodes, grad_clipping=self.GRAD_CLIP, only_return_final=True)
        answer_decoder = lasagne.layers.DenseLayer(brain_layer, num_units=vocab_size, W=lasagne.init.Normal(std=0.1), nonlinearity=lasagne.nonlinearities.softmax)
            
        print("99")
        #l_pred = lasagne.layers.DenseLayer(brain_layer, self.num_classes, W=lasagne.init.Normal(std=0.1), nonlinearity=lasagne.nonlinearities.softmax)
        probas = lasagne.layers.get_output(answer_decoder)  # Get handle on the network
        print("101")
        
        # Building the cost model and Thenao functions
        probas = T.clip(probas, 1e-7, 1.0-1e-7)
        pred = T.argmax(probas, axis=1)
        #cost = T.nnet.binary_crossentropy(probas, y).sum()
        predicted_values = lasagne.layers.get_output(answer_decoder).flatten()
        cost = T.mean((predicted_values - y)**2)
        test_prediction = lasagne.layers.get_output(answer_decoder, deterministic=True)

        params = lasagne.layers.get_all_params(answer_decoder, trainable=True)
        grads = T.grad(cost, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)

        print("314")
        #updates = lasagne.updates.sgd(scaled_grads, params, learning_rate=self.lr) # Note, the scaled grads was causing an error, it was not learning.
        updates = lasagne.updates.adagrad(cost, params, self.LEARNING_RATE)
        
        print(" about to compile")
        assert(1==2)
        # Input to function:
        self.train_model = theano.function([l_in.input_var, y, l_mask.input_var], cost, updates=updates)
        self.compute_cost = theano.function([l_in.input_var, y, l_mask.input_var], cost)
        self.compute_pred = theano.function([l_in.input_var, l_mask.input_var], test_prediction)

        print(" finished compilation: ")


        
        
        
        # TODO:  Test that any of the above even runs
#         l_recurrent_first_seq = lasagne.layers.GRULayer(l_in, self.N_HIDDEN, mask_input=l_mask,grad_clipping=self.GRAD_CLIP)  # output size is (1 = num_batches, 9 = seq_len, 100 = hidden units)
#         l_recurrent_second_seq = lasagne.layers.GRULayer(l_in, self.N_HIDDEN, mask_input=l_mask, grad_clipping=self.GRAD_CLIP)
# 
#         # l_forward_slice = lasagne.layers.SliceLayer(l_recurrent_first_seq, -1, 1)  # INSERTED
# 
#         #self.num_classes = self.vocab_size
#         #l_pred = lasagne.layers.DenseLayer(l_recurrent_first_seq, num_units=self.num_classes, W=lasagne.init.Normal(std=0.1), nonlinearity=lasagne.nonlinearities.softmax)
# 
# 
#         l_out_first_seq = lasagne.layers.DenseLayer(l_recurrent_first_seq, num_units=max_seqlen * self.N_HIDDEN_BRAIN, W=lasagne.init.Normal(std=0.1), nonlinearity=lasagne.nonlinearities.softmax)  # Size of these is (batch_size = 1, seq_len = 9)
#         l_out_second_seq = lasagne.layers.DenseLayer(l_recurrent_first_seq, num_units=max_seqlen * self.N_HIDDEN_BRAIN, W=lasagne.init.Normal(std=0.1), nonlinearity=lasagne.nonlinearities.softmax)  # Could just try to have this have more units
# 
#         l_out_first_seq = lasagne.layers.ReshapeLayer(l_out_first_seq, shape=(self.N_BATCH, self.N_HIDDEN_BRAIN, max_seqlen))
#         l_out_second_seq = lasagne.layers.ReshapeLayer(l_out_second_seq, shape=(self.N_BATCH, self.N_HIDDEN_BRAIN, max_seqlen))
# 
#         input_to_brain = lasagne.layers.ConcatLayer([l_out_first_seq, l_out_second_seq], axis=2)
# 
#         brain_layer = lasagne.layers.GRULayer(input_to_brain, self.N_HIDDEN_BRAIN)
#         # l_forward_slice = lasagne.layers.SliceLayer(brain_layer, -1, 1) # Output to dense softmax layer
#         #
#         self.num_classes = self.vocab_size
#         l_pred = lasagne.layers.DenseLayer(brain_layer, self.num_classes, W=lasagne.init.Normal(std=0.1), nonlinearity=lasagne.nonlinearities.softmax)

        

    def train(self):
        epoch = 0
        n_train_batches = len(self.y_train) // self.N_BATCH
        self.lr = self.init_lr
        prev_train_f1 = None

        print(" training... ")

        try:
            while(epoch < self.num_epochs):
                epoch += 1

                if epoch % 25 == 0:
                    self.lr /= 2.0
                indices = range(n_train_batches)

                total_cost = 0
                start_time = time.time()

                for (x, y, m) in zip(self.X_train, self.y_train, self.mask_train):
                    x = np.array([x.toarray()])
                    m = np.array([m])
                    self.train_model(x, y, m)

                cost_val = 0
                num_correct, total_seen = 0, 0

                for (x, y, m, idx) in zip(self.X_test, self.y_test, self.mask_test, range(len(self.X_test))):

                    x = np.array([x.toarray()])
                    m = np.array([m])
                    cost_val += self.compute_cost(x, y, m)

                    #print(" this is pred: ", self.compute_pred(x, m))
                    #print(" answer: ", self.idx2word[np.argmax(y)], " and prediction: ", self.idx2word[np.argmax(self.compute_pred(x, m)[0])])
                    if self.idx2word[np.argmax(y)] == self.idx2word[np.argmax(self.compute_pred(x, m)[0])]:
                        num_correct += 1
                    total_seen += 1

                print(" Epoch Number: ", epoch, " Percent Correct ", num_correct / total_seen, " cur cost: ", cost_val)

        except KeyboardInterrupt:
            pass



















