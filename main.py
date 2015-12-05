import numpy as np
import theano
import theano.tensor as T
import lasagne

from Preprocessor import Preprocessor
from lstm import LSTM
from mem_network import MemNet
from wqa_processor import WikiProcessor
from cnn_processor import CNNProcessor
from dynamic_mem_net import DynamicMemNet
from babi_processor import BabiProcessor
import sys

def main(nn_type, data_type):

    print(" Starting... ")
    filename_train = 'qa1_single-supporting-fact_train.txt'
    filename_test = 'qa1_single-supporting-fact_test.txt'
    directory = 'data/babi_tasks/tasks_1-20_v1-2/en/'
    num_epochs = 500

    #processor = Preprocessor(directory, filename_train, filename_test, data_type)
    #X_train, y_train, mask_train, X_test, y_test, mask_test, input_size, max_seq_len, idx2word = processor.extract_data()

    #wProc = WikiProcessor('C:/Users/Dan/Desktop/Crore/6.864/Project/Data/wiki_qa/')
    #wProc.process()

    #proc = CNNProcessor()
    # proc.process()


    if nn_type == "lstm":
        proc = BabiProcessor(data_type)
        X_train, y_train, mask_train, X_test, y_test, mask_test, input_size, max_seq_len, idx2word = proc.process()
        lstm = LSTM(X_train, y_train, mask_train, X_test, y_test, mask_test, idx2word)
        network, l_mask, l_in = lstm.build_model(input_size, max_seq_len)
        lstm.optimize(network, l_mask, l_in)

    elif nn_type == "mem_net" and data_type == "babi":
        mn = MemNet()
        mn.run('babi')
    elif nn_type == "mem_net" and data_type == "wiki_qa":
        mn = MemNet()
        mn.run(data_type)
    elif nn_type == "mem_net" and data_type == "cnn":
        mn = MemNet()
        mn.run('cnn_qa')

    elif nn_type == "dynam_net":
        proc = BabiProcessor(data_type)
        X_train, y_train, mask_train, X_test, y_test, mask_test, input_size, max_seq_len, idx2word = proc.process()
        dn = DynamicMemNet(X_train, y_train, mask_train, X_test, y_test, mask_test, input_size, max_seq_len, idx2word)
        dn.build()
        dn.train()

if __name__ == '__main__':
    # See function train for all possible parameter and there definition.

    args = sys.argv[1:]

    if len(args) == 2:
        nn_type = args[0]
        data_type = args[1]
        print(" data type: ", data_type)
    else:
        nn_type = "mem_net"
        data_type = "cnn"


    main(nn_type, data_type)


