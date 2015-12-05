from lasagne import nonlinearities
__author__ = 'Dan'
import numpy as np
import theano 
import theano.tensor as T
import theano.typed_list
import lasagne
import time
from lasagne.layers import MergeLayer
from lasagne.layers import Gate
import lasagne.nonlinearities


class DMNLayerV2(MergeLayer):
    r"""
    lasagne.layers.recurrent.GRULayer(incoming, num_units,
    resetgate=lasagne.layers.Gate(W_cell=None),
    updategate=lasagne.layers.Gate(W_cell=None),
    hidden_update=lasagne.layers.Gate(
    W_cell=None, lasagne.nonlinearities.tanh),
    hid_init=lasagne.init.Constant(0.), backwards=False, learn_init=False,
    gradient_steps=-1, grad_clipping=0, unroll_scan=False,
    precompute_input=True, mask_input=None, only_return_final=False, **kwargs)
    Gated Recurrent Unit (GRU) Layer
    Implements the recurrent step proposed in [1]_, which computes the output
    by
    .. math ::
        r_t &= \sigma_r(x_t W_{xr} + h_{t - 1} W_{hr} + b_r)\\
        u_t &= \sigma_u(x_t W_{xu} + h_{t - 1} W_{hu} + b_u)\\
        c_t &= \sigma_c(x_t W_{xc} + r_t \odot (h_{t - 1} W_{hc}) + b_c)\\
        h_t &= (1 - u_t) \odot h_{t - 1} + u_t \odot c_t
    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
    num_units : int
        Number of hidden units in the layer.
    resetgate : Gate
        Parameters for the reset gate (:math:`r_t`): :math:`W_{xr}`,
        :math:`W_{hr}`, :math:`b_r`, and :math:`\sigma_r`.
    updategate : Gate
        Parameters for the update gate (:math:`u_t`): :math:`W_{xu}`,
        :math:`W_{hu}`, :math:`b_u`, and :math:`\sigma_u`.
    hidden_update : Gate
        Parameters for the hidden update (:math:`c_t`): :math:`W_{xc}`,
        :math:`W_{hc}`, :math:`b_c`, and :math:`\sigma_c`.
    hid_init : callable, np.ndarray, theano.shared or TensorVariable
        Initializer for initial hidden state (:math:`h_0`).  If a
        TensorVariable (Theano expression) is supplied, it will not be learned
        regardless of the value of `learn_init`.
    backwards : bool
        If True, process the sequence backwards and then reverse the
        output again such that the output from the layer is always
        from :math:`x_1` to :math:`x_n`.
    learn_init : bool
        If True, initial hidden values are learned. If `hid_init` is a
        TensorVariable then the TensorVariable is used and
        `learn_init` is ignored.
    gradient_steps : int
        Number of timesteps to include in the backpropagated gradient.
        If -1, backpropagate through the entire sequence.
    grad_clipping : float
        If nonzero, the gradient messages are clipped to the given value during
        the backward pass.  See [1]_ (p. 6) for further explanation.
    unroll_scan : bool
        If True the recursion is unrolled instead of using scan. For some
        graphs this gives a significant speed up but it might also consume
        more memory. When `unroll_scan` is True, backpropagation always
        includes the full sequence, so `gradient_steps` must be set to -1 and
        the input sequence length must be known at compile time (i.e., cannot
        be given as None).
    precompute_input : bool
        If True, precompute input_to_hid before iterating through
        the sequence. This can result in a speedup at the expense of
        an increase in memory usage.
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    only_return_final : bool
        If True, only return the final sequential output (e.g. for tasks where
        a single target value for the entire sequence is desired).  In this
        case, Theano makes an optimization which saves memory.
    References
    ----------
    .. [1] Cho, Kyunghyun, et al: On the properties of neural
       machine translation: Encoder-decoder approaches.
       arXiv preprint arXiv:1409.1259 (2014).
    .. [2] Chung, Junyoung, et al.: Empirical Evaluation of Gated
       Recurrent Neural Networks on Sequence Modeling.
       arXiv preprint arXiv:1412.3555 (2014).
    .. [3] Graves, Alex: "Generating sequences with recurrent neural networks."
           arXiv preprint arXiv:1308.0850 (2013).
    Notes
    -----
    An alternate update for the candidate hidden state is proposed in [2]_:
    .. math::
        c_t &= \sigma_c(x_t W_{ic} + (r_t \odot h_{t - 1})W_{hc} + b_c)\\
    We use the formulation from [1]_ because it allows us to do all matrix
    operations in a single dot product.
    """
    def __init__(self, incoming, question_layer, num_hidden_units_h, num_hidden_units_m, max_seqlen,
                 resetgate_facts=Gate(W_cell=None),
                 updategate_facts=Gate(W_cell=None),
                 hidden_update_facts=Gate(W_cell=None,
                                    nonlinearity=lasagne.nonlinearities.tanh),
                 resetgate_brain=Gate(W_cell=None),
                 updategate_brain=Gate(W_cell=None),
                 hidden_update_brain=Gate(W_cell=None),
                 
                 dmn_gate=Gate(W_cell=None),
                 hid_init=lasagne.init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=True,
                 precompute_input=True,
                 mask_input=None,
                 only_return_final=False,
                 **kwargs):
        
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.  We will just provide the
        # layer input as incomings, unless a mask input was provided.
        incomings = [incoming]
        if mask_input is not None:
            incomings.append(mask_input)
       
        # Initialize parent layer
        super(DMNLayerV2, self).__init__(incomings, **kwargs)
        self.learn_init = learn_init
        self.num_hidden_units_h = num_hidden_units_h
        self.num_hidden_units_m = num_hidden_units_m
        self.grad_clipping = grad_clipping
        self.backwards = backwards
        self.gradient_steps = gradient_steps
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.cur_sequence_idx = 0
        self.max_seqlen = max_seqlen
        self.question_layer = question_layer

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # Input dimensionality is the output dimensionality of the input layer
        num_inputs = np.prod(input_shape[2:])

        def add_gate_params(gate, gate_name, local_num_hidden_units):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            return (self.add_param(gate.W_in, (num_inputs, local_num_hidden_units),
                                   name="W_in_to_{}".format(gate_name)),
                    self.add_param(gate.W_hid, (local_num_hidden_units, local_num_hidden_units),
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (local_num_hidden_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in all parameters from gates
        (self.W_in_to_updategate, self.W_hid_to_updategate, self.b_updategate,
         self.nonlinearity_updategate) = add_gate_params(updategate_facts,
                                                         'updategate', num_hidden_units_h)
        (self.W_in_to_resetgate, self.W_hid_to_resetgate, self.b_resetgate,
         self.nonlinearity_resetgate) = add_gate_params(resetgate_facts, 'resetgate', num_hidden_units_h)

        (self.W_in_to_hidden_update, self.W_hid_to_hidden_update,
         self.b_hidden_update, self.nonlinearity_hid) = add_gate_params(
             hidden_update_facts, 'hidden_update', num_hidden_units_h)

        # These parameters are for the brain GRU
        (self.W_brain_in_to_updategate, self.W_brain_hid_to_updategate, self.b_brain_updategate,
         self.nonlinearity_brain_updategate) = add_gate_params(updategate_brain, 'updategate', num_hidden_units_m)
        
        (self.W_brain_in_to_resetgate, self.W_brain_hid_to_resetgate, self.b_brain_resetgate,
         self.nonlinearity_brain_resetgate) = add_gate_params(resetgate_brain, 'resetgate', num_hidden_units_m)

        (self.W_brain_in_to_hidden_update, self.W_brain_hid_to_hidden_update,
         self.b_brain_hidden_update, self.nonlinearity_brain_hid_update) = add_gate_params(hidden_update_brain, 'hidden_update', num_hidden_units_m)

        size_fact_embedding = 20  # TODO DS: change these from constants, just put here for now for 
        size_question_embedding = size_fact_embedding
        size_dmn_gate_vector = 9
        size_hidden_state = size_fact_embedding
         
        # (self.W_dmn_b, self.W_dmn_1, self.W_dmn_2, self.b_dmn_1, self.b_dmn_2) = add_gate_params(dmn_gate, "dmn_gate")
        self.W_dmn_b = self.add_param(lasagne.init.Normal(0.1), (size_fact_embedding, size_question_embedding) , name="W_dmn_b")
        self.W_dmn_1 = self.add_param(lasagne.init.Normal(0.1), (size_hidden_state, size_dmn_gate_vector), name="W_dmn_1")
        self.W_dmn_2 = self.add_param(lasagne.init.Normal(0.1), (size_hidden_state, size_hidden_state), name="W_dmn_2")
                
        self.b_dmn_1 = self.add_param(lasagne.init.Normal(0.1), (size_hidden_state, 1), name="b_dmn_1")
        self.b_dmn_2 = self.add_param(lasagne.init.Normal(0.1), (size_hidden_state, 1), name="b_dmn_2")  

        # Initialize hidden state
        if isinstance(hid_init, T.TensorVariable):
            if hid_init.ndim != 2:
                raise ValueError(
                    "When hid_init is provided as a TensorVariable, it should "
                    "have 2 dimensions and have shape (num_batch, num_units)")
            self.hid_init = hid_init
        else:
            self.hid_init = self.add_param(
                hid_init, (1, self.num_hidden_units_h + self.num_hidden_units_m), name="hid_init",
                trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_hidden_units_m            
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], self.max_seqlen, self.num_hidden_units_m

    def get_output_for(self, inputs, **kwargs):
        """
        Compute this layer's output function given a symbolic input variable
        Parameters
        ----------
        inputs : list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        Returns
        -------
        layer_output : theano.TensorType
            Symbolic output variable.
        """
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = inputs[1] if len(inputs) > 1 else None

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 3*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_resetgate, self.W_in_to_updategate,
             self.W_in_to_hidden_update], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_resetgate, self.W_hid_to_updategate,
             self.W_hid_to_hidden_update], axis=1)

        # Stack gate biases into a (3*num_units) vector
        b_stacked = T.concatenate(
            [self.b_resetgate, self.b_updategate,
             self.b_hidden_update], axis=0)
        
        # Stacking for brain layer
        W_brain_in_stacked = T.concatenate(
            [self.W_brain_in_to_resetgate, self.W_brain_in_to_updategate,
             self.W_brain_in_to_hidden_update], axis=1)
        W_brain_hid_stacked = T.concatenate(
            [self.W_brain_hid_to_resetgate, self.W_brain_hid_to_updategate,
             self.W_brain_hid_to_hidden_update], axis=1)
        b_brain_stacked = T.concatenate(
            [self.b_brain_resetgate, self.b_brain_updategate,
             self.b_brain_hidden_update], axis=0)
                

        if self.precompute_input:
            # precompute_input inputs*W. W_in is (n_features, 3*num_units).
            # input is then (n_batch, n_time_steps, 3*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 3*num_units).
        # We define a slicing function that extract the input to each GRU gate
        def slice_w_h(x, n):
            return x[:, n * self.num_hidden_units_h:(n + 1) * self.num_hidden_units_h]

        def slice_w_m(x, n):
            return x[:, n * self.num_hidden_units_m:(n + 1) * self.num_hidden_units_m]

        # Create single recurrent computation step function
        # input__n is the n'th vector of the input
        def step(input_n, hid_previous_total, *args):
            
            hid_previous_facts = hid_previous_total[0:self.num_hidden_units_h]
            hid_previous_brain = hid_previous_total[self.num_hidden_units_h:]
            
            self.cur_sequence_idx += 1  # Updates where we are at in the sequence
            
            # Compute W_{hr} h_{t - 1}, W_{hu} h_{t - 1}, and W_{hc} h_{t - 1}
            hid_input_facts = T.dot(hid_previous_facts, W_hid_stacked)


            if self.grad_clipping:
                input_n = theano.gradient.grad_clip(
                    input_n, -self.grad_clipping, self.grad_clipping)
                hid_input_facts = theano.gradient.grad_clip(
                    hid_input_facts, -self.grad_clipping, self.grad_clipping)

            if not self.precompute_input:
                # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
                input_n = T.dot(input_n, W_in_stacked) + b_stacked  # DS Note:  accomplishes the multiplication AND adds bias

            # Reset and update gates
            resetgate = slice_w_h(hid_input_facts, 0) + slice_w_h(input_n, 0)
            updategate = slice_w_h(hid_input_facts, 1) + slice_w_h(input_n, 1)
            resetgate = self.nonlinearity_resetgate(resetgate)
            updategate = self.nonlinearity_updategate(updategate)
            
            # DS Edit: DynamMemNet modifiers
            m_dmn = hid_previous_brain  # Note that this should have size 
            c_dmn = input_n  # This is a TesnorType<float64, row>
            q_dmn = self.question_layer  # This is a lasagne recurrent GRU layer

            z_dmn = T.concatenate([c_dmn, m_dmn, q_dmn, c_dmn * q_dmn, abs(c_dmn - q_dmn), abs(c_dmn - m_dmn), T.dot(c_dmn.T, T.dot(self.W_dmn_b, q_dmn)),
                         T.dot(c_dmn.T, T.dot(self.W_dmn_b, m_dmn))], axis=1)
            G_dmn = nonlinearities.sigmoid(T.dot(self.W_dmn_2, nonlinearities.tanh(T.dot(self.W_dmn_1, z_dmn)) + self.b_dmn_1) + self.b_dmn_2)
            # Note, you also need W_b for the c and q elements.
            #something_else = T.dot(hid_previous_facts, W_hid_stacked)
            hidden_update_in = slice_w_h(input_n, 2)
            hidden_update_hid = slice_w_h(hid_input_facts, 2)
            hidden_update_facts = hidden_update_in + resetgate * hidden_update_hid
            if self.grad_clipping:
                hidden_update_facts = theano.gradient.grad_clip(
                    hidden_update_facts, -self.grad_clipping, self.grad_clipping)
            hidden_update_facts = self.nonlinearity_hid(hidden_update_facts)

            # Compute (1 - u_t)h_{t - 1} + u_t c_t
            hid = (1 - updategate) * hid_previous_facts + updategate * hidden_update_facts  # This is the GRU_fact output
            #output_dmn = G_dmn * hid + (1 - G_dmn) * hid_previous_facts  # This is the output of the Dynamic Memory Net modified GRU, Eq. (5)
            output_dmn = hid
                        
#             if self.cur_sequence_idx == self.max_seqlen:
#                 hid_input_brain = T.dot(hid_previous_brain, W_brain_hid_stacked)            
#             
#                 if self.grad_clipping:
#                     input_to_brain = theano.gradient.grad_clip(
#                         output_dmn, -self.grad_clipping, self.grad_clipping)
#                     hid_input_brain = theano.gradient.grad_clip(
#                         hid_input_brain, -self.grad_clipping, self.grad_clipping)
#                 else:
#                     input_to_brain = output_dmn
#                     
#                 if not self.precompute_input:
#                     # Compute W_{xr}x_t + b_r, W_{xu}x_t + b_u, and W_{xc}x_t + b_c
#                     input_to_brain = T.dot(input_to_brain, W_brain_in_stacked) + b_brain_stacked  # DS Note:  accomplishes the multiplication AND adds bias
#             
#                 # Reset and update gates
#                 resetgate_brain = slice_w_m(hid_input_brain, 0) + slice_w_m(input_to_brain, 0)
#                 updategate_brain = slice_w_m(hid_input_brain, 1) + slice_w_m(input_to_brain, 1)
#                 resetgate_brain = self.nonlinearity_brain_resetgate(resetgate_brain)
#                 updategate_brain = self.nonlinearity_brain_updategate(updategate_brain)
#             
#                 hidden_update_in_brain = slice_w_m(input_to_brain, 2)
#                 hidden_update_brain = slice_w_m(hid_input_brain, 2)
#                 hidden_update_brain = hidden_update_in_brain + resetgate_brain * hidden_update_brain
#                 
#                 if self.grad_clipping:
#                     hidden_update_brain = theano.gradient.grad_clip(hidden_update_brain, -self.grad_clipping, self.grad_clipping)
#                 hidden_update_brain = self.nonlinearity_brain_hid_update(hidden_update_brain)
#                 
#                 hid_brain = (1 - updategate_brain) * hid_previous_brain + updategate_brain * hidden_update_brain
#             
#             else:                
#             
            hid_brain = hid_previous_brain
                              
            return T.concatenate([output_dmn, hid_brain], axis=1)
            
        def step_masked(input_n, mask_n, hid_previous, *args):
            hid = step(input_n, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            not_mask = 1 - mask_n
            hid = hid*mask_n + hid_previous*not_mask

            return hid

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = [input]
            step_fun = step

        if isinstance(self.hid_init, T.TensorVariable):
            hid_init = self.hid_init
        else:
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            hid_init = T.dot(T.ones((num_batch, 1)), self.hid_init)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked, W_brain_in_stacked, W_brain_hid_stacked, self.W_dmn_1, self.W_dmn_2, self.W_dmn_b]
        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        #if self.unroll_scan:
        if 1 == 2:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            hid_out = lasagne.utils.unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])[0]
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            print("442")
            hid_out, self.theano_updates = theano.scan(
                fn=step_fun,
                sequences=sequences,
                go_backwards=self.backwards,
                outputs_info=[hid_init],
                non_sequences=non_seqs,
                truncate_gradient=self.gradient_steps,
                strict=True)[0]
            print(" Got past 450")

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            hid_out = hid_out.dimshuffle(1, 0, 2)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, ::-1]

        return hid_out