import numpy as np
import theano
import theano.tensor as T
import lasagne

class LSTM(object):

    N_BATCH = 1
    N_HIDDEN = 50
    GRAD_CLIP = 100
    LEARNING_RATE = .001

    def __init__(self, X_train, y_train, mask_train, X_test, y_test, mask_test, idx2word):
        self.X_train = X_train
        self.y_train = y_train
        self.mask_train = mask_train
        self.X_test = X_test
        self.y_test = y_test
        self.mask_test = mask_test
        self.idx2word = idx2word

        self.num_epochs = 500

    def build_model(self, input_size, max_seq_len, input_var=None):
        num_units = 5
        num_classes = input_size

        # l_inp = lasagne.layers.InputLayer((None, None, input_size))
        l_in = lasagne.layers.InputLayer(shape=(self.N_BATCH, max_seq_len, input_size))
        l_mask = lasagne.layers.InputLayer(shape=(self.N_BATCH, max_seq_len))

        #l_recurrent = lasagne.layers.LSTMLayer(l_in, self.N_HIDDEN, mask_input=l_mask, grad_clipping=self.GRAD_CLIP, nonlinearity=lasagne.nonlinearities.tanh)

        l_recurrent = lasagne.layers.GRULayer(l_in, self.N_HIDDEN, mask_input=l_mask, grad_clipping=self.GRAD_CLIP)

        l_out = lasagne.layers.DenseLayer(l_recurrent, num_units=input_size, nonlinearity=lasagne.nonlinearities.softmax)

        #print(" l recurrent size: ", lasagne.layers.get_output_shape(l_recurrent))  # This is (1, 9, 50)
        #print(" l dense: shape: ", lasagne.layers.get_output_shape(l_out)) # This is (1 , 19)

        # batchsize, seqlen, _ = l_inp.input_var.shape
        #
        # l_lstm = lasagne.layers.LSTMLayer(l_inp, num_units=num_units)
        # l_dense = lasagne.layers.DenseLayer(l_lstm, num_units=num_classes)

        return l_out, l_mask, l_in


    def optimize(self, network, l_mask, l_in):

        target_values = T.vector('target_output')

        network_output = lasagne.layers.get_output(network)
        test_prediction = lasagne.layers.get_output(network, deterministic=True)

        predicted_values = network_output.flatten()

        cost = T.mean((predicted_values - target_values)**2)
        # Retrieve all parameters from the network
        all_params = lasagne.layers.get_all_params(network)
        # Compute SGD updates for training
        print("Computing updates ...")
        updates = lasagne.updates.adagrad(cost, all_params, self.LEARNING_RATE)
        # Theano functions for training and computing cost
        print("Compiling functions ...")


        train = theano.function([l_in.input_var, target_values, l_mask.input_var], cost, updates=updates)
        compute_cost = theano.function([l_in.input_var, target_values, l_mask.input_var], cost)
        gen_prediction = theano.function([l_in.input_var, l_mask.input_var], test_prediction)

        print("Training...")

        # print(" this is x test: ", len(X_test))
        # print(" y test shape: ", len(y_test))
        # print(" and mask shape: ", len(mask_test))

        try:
            for epoch in range(self.num_epochs):

                for (x, y, m) in zip(self.X_train, self.y_train, self.mask_train):
                    x = np.array([x.toarray()])
                    m = np.array([m])
                    train(x, y, m)

                cost_val = 0
                rand_test_sample = np.random.randint(len(self.X_test), size=1)

                num_correct, total_seen = 0, 0

                for (x, y, m, idx) in zip(self.X_test, self.y_test, self.mask_test, range(len(self.X_test))):
                    x = np.array([x.toarray()])
                    m = np.array([m])

                    cost_val += compute_cost(x, y, m)
                    #print("89, cost: ", cost_val)
                    # test_prediction = lasagne.layers.get_output(network, deterministic=True)

                    # whoknowswhat = gen_prediction(x, m)
                    # if rand_test_sample == idx:
                    #     print(" sentence: ", self.idx_sentence_to_string(x, self.idx2word))
                    #     print(" who knows what: ", self.idx2word[np.argmax(whoknowswhat[0])])
                    #     print(" this is y: ", y)
                    #     print(" actual value: ", self.idx2word[np.argmax(y)])

                    if self.idx2word[np.argmax(y)] == self.idx2word[np.argmax(gen_prediction(x, m)[0])]:
                        num_correct += 1
                    total_seen +=1

                    #predict_fn = theano.function([x], T.argmax(test_prediction, axis=1))
                    #print("Predicted class for first test input: %r" % predict_fn)

                print(" Epoch: " , epoch, " ratio correct: ", num_correct / total_seen)

                print(" Epoch: ", epoch, " cost val: ", cost_val)

        except KeyboardInterrupt:
            pass


    def idx_sentence_to_string(self, x, idx2word):
        cur_sentence = ""
        for idx in x[0]:
            if np.max(idx) > 0:
                cur_sentence += " " + idx2word[np.argmax(idx)]
        return cur_sentence

