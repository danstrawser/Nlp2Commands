__author__ = 'Dan'
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
    N_HIDDEN_BRAIN = 20
    GRAD_CLIP = 100
    LEARNING_RATE = .001


    def __init__(self, X_train, y_train, mask_train, X_test, y_test, mask_test, input_size, max_seq_len, idx2word):
        
        self.X_train = X_train
        self.y_train = y_train
        self.mask_train = mask_train
        self.X_test = X_test
        self.y_test = y_test
        self.mask_test = mask_test
        self.input_size = input_size
        self.max_seq_len = max_seq_len
        self.idx2word = idx2word
        self.vocab_size = len(idx2word)

        self.max_norm = 40      # Same as the MemNet
        self.init_lr = .001     # Same as the MemNet
        self.lr= self.init_lr   # Same as the MemNet
        self.num_epochs = 100

    # Note that you could want to create an embedding for input context
    def build(self, input_var=None):
        print(" Initializing Dynamic Mem Net with Learning Rate: ", self.LEARNING_RATE)
        input_size, max_seqlen = self.input_size, self.max_seq_len

        y = T.vector('target_output')
        # Build the network
        l_in = lasagne.layers.InputLayer(shape=(self.N_BATCH, max_seqlen, input_size))
        l_mask = lasagne.layers.InputLayer(shape=(self.N_BATCH, max_seqlen))
        l_recurrent_first_seq = lasagne.layers.GRULayer(l_in, self.N_HIDDEN, mask_input=l_mask,grad_clipping=self.GRAD_CLIP)  # output size is (1 = num_batches, 9 = seq_len, 100 = hidden units)
        l_recurrent_second_seq = lasagne.layers.GRULayer(l_in, self.N_HIDDEN, mask_input=l_mask, grad_clipping=self.GRAD_CLIP)

        # l_forward_slice = lasagne.layers.SliceLayer(l_recurrent_first_seq, -1, 1)  # INSERTED

        #self.num_classes = self.vocab_size
        #l_pred = lasagne.layers.DenseLayer(l_recurrent_first_seq, num_units=self.num_classes, W=lasagne.init.Normal(std=0.1), nonlinearity=lasagne.nonlinearities.softmax)


        l_out_first_seq = lasagne.layers.DenseLayer(l_recurrent_first_seq, num_units=max_seqlen * self.N_HIDDEN_BRAIN, W=lasagne.init.Normal(std=0.1), nonlinearity=lasagne.nonlinearities.softmax)  # Size of these is (batch_size = 1, seq_len = 9)
        l_out_second_seq = lasagne.layers.DenseLayer(l_recurrent_first_seq, num_units=max_seqlen * self.N_HIDDEN_BRAIN, W=lasagne.init.Normal(std=0.1), nonlinearity=lasagne.nonlinearities.softmax)  # Could just try to have this have more units

        l_out_first_seq = lasagne.layers.ReshapeLayer(l_out_first_seq, shape=(self.N_BATCH, self.N_HIDDEN_BRAIN, max_seqlen))
        l_out_second_seq = lasagne.layers.ReshapeLayer(l_out_second_seq, shape=(self.N_BATCH, self.N_HIDDEN_BRAIN, max_seqlen))

        input_to_brain = lasagne.layers.ConcatLayer([l_out_first_seq, l_out_second_seq], axis=2)

        brain_layer = lasagne.layers.GRULayer(input_to_brain, self.N_HIDDEN_BRAIN)
        # l_forward_slice = lasagne.layers.SliceLayer(brain_layer, -1, 1) # Output to dense softmax layer
        #
        self.num_classes = self.vocab_size
        l_pred = lasagne.layers.DenseLayer(brain_layer, self.num_classes, W=lasagne.init.Normal(std=0.1), nonlinearity=lasagne.nonlinearities.softmax)

        #probas = lasagne.layers.get_output(l_pred)  # Get handle on the network

        # Building the cost model and Thenao functions
        #probas = T.clip(probas, 1e-7, 1.0-1e-7)
        #pred = T.argmax(probas, axis=1)
        #cost = T.nnet.binary_crossentropy(probas, y).sum()
        predicted_values = lasagne.layers.get_output(l_pred).flatten()
        cost = T.mean((predicted_values - y)**2)
        test_prediction = lasagne.layers.get_output(l_pred, deterministic=True)

        params = lasagne.layers.get_all_params(l_pred, trainable=True)
        grads = T.grad(cost, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads, self.max_norm)

        #updates = lasagne.updates.sgd(scaled_grads, params, learning_rate=self.lr) # Note, the scaled grads was causing an error, it was not learning.
        updates = lasagne.updates.adagrad(cost, params, self.LEARNING_RATE)

        # Input to function:
        self.train_model = theano.function([l_in.input_var, y, l_mask.input_var], cost, updates=updates)
        self.compute_cost = theano.function([l_in.input_var, y, l_mask.input_var], cost)
        self.compute_pred = theano.function([l_in.input_var, l_mask.input_var], test_prediction)

        print(" finished compilation: ")


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


