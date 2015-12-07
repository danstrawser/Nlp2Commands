import theano
import theano.tensor as T
import numpy as np
import time

use_scan = True # change this to use the unrolled expression

m = 2**10 # batch size
n = 2**12 # number of hidden units per layer
depth = 8
t = 8 # time steps

assert theano.config.floatX == 'float32'
assert theano.config.optimizer == 'fast_run'
np.random.seed(0)

def relu(x): return x * (x > 0)
def rand(*size): return np.array(np.random.normal(size=size, scale=1e-3), dtype=theano.config.floatX)
def init(*size): return theano.shared(rand(*size))

print("symbolic input...")
x = T.tensor3()
targets = T.tensor3()
w = T.matrix()
g_out = T.matrix()
h0 = [T.matrix() for i in range(depth)]

def rnn_step(*args):
    x_curr  = args[0]
    h_prev  = args[1:]
    h_curr = []
    for h in h_prev:
        h_below = x_curr if len(h_curr) == 0 else h_curr[-1]
        h_curr += [relu(T.dot(h_below, w) + T.dot(h, w))]
    return h_curr

if use_scan:
    out, updates = theano.scan(rnn_step,
                               sequences=x,
                               outputs_info=h0,
                               non_sequences=[])
    err = ((out[-1] - targets) ** 2).mean()
else:
    err = 0
    state = h0
    for i in range(t):
        # I think this line concatenates the hidden state and input:
        args = [x[i]] + state  # x is the tensor3, state is first the initial state, then updated by the RNN
        state = rnn_step(*args)
        err += ((state[-1] - targets[i]) ** 2).mean() / t
    updates = theano.OrderedUpdates()

print("allocating...")
g_out = init(n, n)  # initiates shared variable with size number of units per hidden layer
x_val = rand(t, m, n)  # initialize data random, with t=seq_len, m=batch_size, n=hidden_units
targets_val = rand(t, m, n)
w_val = rand(n, n)
h0_val = [rand(m, n) for i in range(depth)] # I believe this is depth per layer

print("compiling...")
f = theano.function([x, w, targets] + h0, err, updates=updates + [(g_out, T.grad(err, w))])

t0 = time.time()
print("running...")
f_out = f(x_val, w_val, targets_val, *h0_val)
elapsed= time.time() - t0

print(f_out, g_out.get_value())

GB = 4. * (m * n * (depth+2) + m * n * depth + 2 * n * n * depth) / 1024**3
TFLOPS = 3 * 2 * 2 * m * n * n * t * depth / (time.time() - t0) / 1e12

print("expected memory usage =", GB, "GB, measured TFLOPS =", TFLOPS)