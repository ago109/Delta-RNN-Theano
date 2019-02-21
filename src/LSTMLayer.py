import logging
import numpy
import theano
from theano import config
import theano.tensor as T
import Utils as Util

# Construct a Long-Short-Term-Memory layer
class LSTMLayer:

    def __init__(self, rng, input, mask, user, n_in, n_h, n_u, sd=.005, useLN=False):
        self.useLN = useLN
        # Init params
        self.W_i = theano.shared(Util.gauss_weight(n_in, n_h, sd=sd), 'W_i', borrow=True)
        self.numParams = n_in * n_h
        self.W_f = theano.shared(Util.gauss_weight(n_in, n_h, sd=sd), 'W_f', borrow=True)
        self.numParams += n_in * n_h
        self.W_c = theano.shared(Util.gauss_weight(n_in, n_h, sd=sd), 'W_c', borrow=True)
        self.numParams += n_in * n_h
        self.W_o = theano.shared(Util.gauss_weight(n_in, n_h, sd=sd), 'W_o', borrow=True)
        self.numParams += n_in * n_h

        self.U_i = theano.shared(Util.gauss_weight(n_h, sd=sd), 'U_i', borrow=True)
        self.numParams += n_h * n_h
        self.U_f = theano.shared(Util.gauss_weight(n_h, sd=sd), 'U_f', borrow=True)
        self.numParams += n_h * n_h
        self.U_c = theano.shared(Util.gauss_weight(n_h, sd=sd), 'U_c', borrow=True)
        self.numParams += n_h * n_h
        self.U_o = theano.shared(Util.gauss_weight(n_h, sd=sd), 'U_o', borrow=True)
        self.numParams += n_h * n_h

        self.b_i = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_i', borrow=True)
        self.numParams += n_h
        self.b_f = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                 'b_f', borrow=True) # TRICK: init forget-gate to 1 to improve performance!
        self.numParams += n_h
        self.b_c = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_c', borrow=True)
        self.numParams += n_h
        self.b_o = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_o', borrow=True)
        self.numParams += n_h

        self.Q_i = theano.shared(Util.gauss_weight(n_u, n_h, sd=sd), 'Q_i', borrow=True)
        self.numParams += n_u * n_h

        if self.useLN: # optional layer normalization
            self.bi_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'bi_ln', borrow=True)
            self.si_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'si_ln', borrow=True)
            self.bf_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'bi_ln', borrow=True)
            self.sf_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'si_ln', borrow=True)
            self.bc_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'bi_ln', borrow=True)
            self.sc_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'si_ln', borrow=True)
            self.bo_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'bi_ln', borrow=True)
            self.so_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'si_ln', borrow=True)

            self.numParams += (n_h * 2)

        self.params = [self.W_i, self.W_f, self.W_c, self.W_o, self.Q_i,
                       self.U_i, self.U_f, self.U_c, self.U_o,
                       self.b_i, self.b_f, self.b_c, self.b_o]

        outputs_info = [T.zeros((input.shape[1], n_h)),
                        T.zeros((input.shape[1], n_h))]

        rval, updates = theano.scan(self._step,
                                    sequences=[mask, input, user],
                                    outputs_info=outputs_info)

        # self.output is in the format (batchsize, n_h)
        self.output = rval[0]

    # layer normalization
    def ln(self, x, b, s):
        _eps = 1e-5
        output = (x - x.mean(1)[:,None]) / T.sqrt((x.var(1)[:,None] + _eps))
        output = s[None, :] * output + b[None,:]
        return output

    def _step(self, m_, x_, u_, h_, c_):

        uu = (Util.index_dot(u_, self.Q_i)) # share user-weights across gates

        i_preact = (Util.index_dot(x_, self.W_i) +
                    T.dot(h_, self.U_i) + self.b_i + uu)
        if self.useLN: # apply layer normalization to pre-activtions (inner fun)
            i_preact =  self.ln(i_preact, self.b_ln, self.s_ln)
        i = T.nnet.sigmoid(i_preact)

        f_preact = (Util.index_dot(x_, self.W_f) +
                    T.dot(h_, self.U_f) + self.b_f + uu)
        f = T.nnet.sigmoid(f_preact)

        o_preact = (Util.index_dot(x_, self.W_o) +
                    T.dot(h_, self.U_o) + self.b_o + uu)
        o = T.nnet.sigmoid(o_preact)

        c_preact = (Util.index_dot(x_, self.W_c) +
                    T.dot(h_, self.U_c) + self.b_c + uu)
        c = T.tanh(c_preact)

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * T.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

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
