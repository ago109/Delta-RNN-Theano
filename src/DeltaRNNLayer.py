import logging
import numpy
import theano
from theano import config
import theano.tensor as T
import theano.tensor.shared_randomstreams
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import Utils as Util

# Construct the proposed MI-Delta-RNN layer
class DeltaRNNLayer:
    """Delta-RNN (full) (Ororbia et al., 2017) - Learning Simpler Language Models with the Differential State Framework

    This is a concrete instantiation of the general framework proposed in:
    https://arxiv.org/abs/1703.08864
    shown to unify all gated neural architectures and was shown to do the
    roughly the same thing as LSTMs and GRUs with a far simpler architecture
    and an intuitive interpretation from surprisal theory.

    @author - Alexander G. Ororbia II
    """

    def __init__(self, rng, input, mask, dmask, n_in, n_h, sd=.005, useLN=False, drop_p=0.0, funtype="identity", useFanInFanOut=0, useOrtho=0, drop_inner=0):
        # Init params
        self.rng = rng
        self.drop_p = drop_p
        self.useLN = useLN
        self.funtype = funtype
        self.drop_inner = drop_inner
        # Inner function g(.) parameters
        if useFanInFanOut == 1:
            print(" Using fan-in-fan-out init for recurrent params!")
            self.W_i = theano.shared(Util.fanInFanOut((n_in, n_h)), 'W_i', borrow=True)
            self.U_i = theano.shared(Util.fanInFanOut((n_h, n_h)), 'U_i', borrow=True)
        elif useOrtho == 1:
            print(" Using orthogonal init for recurrent params!")
            self.W_i = theano.shared(Util.rand_ortho_u((n_in, n_h)), 'W_i', borrow=True)
            self.U_i = theano.shared(Util.rand_ortho_u((n_h, n_h)), 'U_i', borrow=True)
        else:
            self.W_i = theano.shared(Util.gauss_weight(n_in, n_h, sd=sd), 'W_i', borrow=True)
            self.U_i = theano.shared(Util.gauss_weight(n_h, sd=sd), 'U_i', borrow=True)
        self.numParams = n_in * n_h
        self.numParams += n_h * n_h
        self.numParams += n_h

        if self.useLN: # optional layer normalization
            self.bx_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'bx_ln', borrow=True)
            self.sx_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'sx_ln', borrow=True)
            self.br_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'br_ln', borrow=True)
            self.sr_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'sr_ln', borrow=True)
            self.bu_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'bu_ln', borrow=True)
            self.su_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'su_ln', borrow=True)
            self.numParams += (n_h * 6)
            self.b_r = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'b_r', borrow=True)
            self.numParams += n_h
            self.params = [self.W_i, self.U_i, self.bx_ln, self.sx_ln,
                           self.br_ln, self.sr_ln, self.bu_ln, self.su_ln, self.b_r]
        else:
            #self.alpha = theano.shared(numpy.full((n_h,), 2.0, dtype=config.floatX),
            #                         'alpha', borrow=True) # <-- this was used for PTB in original paper
            self.alpha = theano.shared(numpy.full((n_h,), 1.0, dtype=config.floatX),
                                     'alpha', borrow=True)
            self.numParams += n_h
            self.beta0 = theano.shared(numpy.full((n_h,), 1.0, dtype=config.floatX),
                                     'beta0', borrow=True)
            self.numParams += n_h
            self.beta1 = theano.shared(numpy.full((n_h,), 1.0, dtype=config.floatX),
                                     'beta1', borrow=True)
            self.b_i = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'b_i', borrow=True)
            self.numParams += n_h
            self.b_r = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'b_r', borrow=True)
            self.numParams += n_h
            self.params = [self.W_i, self.U_i, self.b_i, self.b_r,
                           self.alpha, self.beta0, self.beta1]

        outputs_info = [T.zeros((input.shape[1], n_h))]

        rval, updates = theano.scan(self._step,
                                    sequences=[mask, input, dmask],
                                    outputs_info=outputs_info)

        # self.output is in the format (batchsize, n_h)
        self.output = rval #[0]

    # layer normalization
    def ln(self, x, b, s):
        _eps = 1e-5
        output = (x - x.mean(1)[:,None]) / T.sqrt((x.var(1)[:,None] + _eps))
        output = s[None, :] * output + b[None,:]
        return output

    def _step(self, m_, x_, d_, h_):
        # Define Delta-RNN functionality here...
        # d1_t = alpha * V h_t-1 * W x_t
        # d2_t = beta1 * V h_t-1 + beta2 * W x_t
        # g = phi(d1_t + d2_t + b)
        # Compute g(.), the inner function (multiplicative model)
        wx_preact = (Util.index_dot(x_, self.W_i))
        if self.useLN: # apply layer normalization to pre-activtions (inner fun)
            wx_preact = self.ln(wx_preact, self.bx_ln, self.sx_ln)
        uhPrev_preact = (T.dot(h_, self.U_i))
        if self.useLN: # apply layer normalization to pre-activtions (inner fun)
            uhPrev_preact = self.ln(uhPrev_preact, self.br_ln, self.sr_ln)

        # compute multiplicative activations
        if self.useLN:
            d1 = (uhPrev_preact * wx_preact )
            d2 = ((uhPrev_preact) + (wx_preact))
            g_preact = d1 + d2
        else:
            d1 = (self.alpha * uhPrev_preact * wx_preact )
            d2 = ((self.beta0 * uhPrev_preact) + (self.beta1 * wx_preact))
            g_preact = d1 + d2 + self.b_i

        #if self.useLN: # apply layer normalization to pre-activtions (inner fun)
        #    g_preact =  self.ln(g_preact, self.b_ln, self.s_ln)
        g = T.tanh(g_preact)

        if self.drop_inner > 0.0:
            print(" >> Using INNER drop-out")
            if self.drop_p > 0.0: # apply drop-out to post-activations of inner fun
                g = (g * d_) #/(1.0-self.drop_p)
        # Compute f(.), the outer function
        # f = Phi((1-r) g + r (h_t-1))
        # r = sigm(W x_t + b_r)
        #if self.useLN:
        #    r = T.nnet.sigmoid(wx_preact) # calculate data-driven gate
        #else:
        #    r = T.nnet.sigmoid(wx_preact + self.b_r)
        r = T.nnet.sigmoid(wx_preact + self.b_r) # calculate data-driven gate
	if self.funtype == "tanh":
            h = T.tanh(((1. - r) * g) + (r * h_))
        elif self.funtype == "relu":
            h = T.nnet.relu(((1. - r) * g) + (r * h_))
	elif self.funtype == "sigmoid":
            h = T.nnet.sigmoid(((1. - r) * g) + (r * h_))
        else:
            h = (((1. - r) * g) + (r * h_))
        if self.drop_inner == 0.0:
            print(" >> Using OUTER drop-out!")
            if self.drop_p > 0.0: # apply drop-out to post-activations of outer fun
                h = (h * d_) #h = (h / (1.0-self.drop_p)) * d_
        #h = T.tanh(((1. - r) * g) + (r * h_))
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h

    def getNumParams(self): # counts total # params in this model
        return self.numParams

    # Generic saving routine
    def save(self, filename):
        """
        Save the model to file `filename`
        """
        vals = dict([(x.name, x.get_value()) for x in self.params])
        numpy.savez(filename, **vals)

    # Generic loading routine
    def load(self, filename, parameter_strings_to_ignore=[]):
        """
        Load the model.
        Any parameter which has one of the strings inside parameter_strings_to_ignore as a substring,
        will not be loaded from the file (but instead initialized as a new model, which usually means random).
        """
        vals = numpy.load(filename)
        for p in self.params:
            load_parameter = True
            for string_to_ignore in parameter_strings_to_ignore:
                if string_to_ignore in p.name:
                     print('Initializing parameter {} as in new model'.format(p.name))
                     load_parameter = False

            if load_parameter:
                if p.name in vals:
                    print('Loading {} of {}'.format(p.name, p.get_value(borrow=True).shape))
                    if p.get_value().shape != vals[p.name].shape:
                        raise Exception('Shape mismatch: {} != {} for {}'.format(p.get_value().shape, vals[p.name].shape, p.name))
                    p.set_value(vals[p.name])
                else:
                    print('No parameter {} given: default initialization used'.format(p.name))
                    unknown = set(vals.keys()) - {p.name for p in self.params}
                    if len(unknown):
                        print('Unknown parameters {} given'.format(unknown))
