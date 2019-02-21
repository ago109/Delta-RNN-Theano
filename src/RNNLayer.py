import logging
import numpy
import theano
from theano import config
import theano.tensor as T
import theano.tensor.shared_randomstreams
import Utils as Util

# Construct an Elman-RNN layer
class RNNLayer:

    def __init__(self, rng, input, mask, dmask, user, n_in, n_h, n_u, sd=.005, useLN=False, drop_p=0.0):
        # Init params
        self.rng = rng
        self.drop_p = drop_p
        self.useLN = useLN
        self.W_i = theano.shared(Util.gauss_weight(n_in, n_h, sd=sd), 'W_i', borrow=True)
        self.numParams = n_in * n_h
        self.U_i = theano.shared(Util.gauss_weight(n_h, sd=sd), 'U_i', borrow=True)
        self.numParams += n_h * n_h
        self.b_i = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                 'b_i', borrow=True)
        self.numParams += n_h
        self.Q_i = theano.shared(Util.gauss_weight(n_u, n_h, sd=sd), 'Q_i', borrow=True)
        self.numParams += n_u * n_h

        if self.useLN: # optional layer normalization
            self.b_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'b_ln', borrow=True)
            self.s_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     's_ln', borrow=True)
            self.numParams += (n_h * 2)

        if self.useLN:
            self.params = [self.W_i, self.U_i, self.b_i, self.Q_i, self.b_ln, self.s_ln]
        else:
            self.params = [self.W_i, self.U_i, self.b_i, self.Q_i]

        outputs_info = [T.zeros((input.shape[1], n_h))]

        rval, updates = theano.scan(self._step,
                                    sequences=[mask, input, user, dmask],
                                    outputs_info=outputs_info)
        #TODO: truncate_gradient

        # self.output is in the format (batchsize, n_h)
        self.output = rval #[0]

    # layer normalization
    def ln(self, x, b, s):
        _eps = 1e-5
        output = (x - x.mean(1)[:,None]) / T.sqrt((x.var(1)[:,None] + _eps))
        output = s[None, :] * output + b[None,:]
        return output

    def _step(self, m_, x_, u_, d_, h_):
        # Compute g(.), since RNN only has an inner function (no outer)
        user_preact = (Util.index_dot(u_, self.Q_i))
        wx_preact = (Util.index_dot(x_, self.W_i))
        uhPrev_preact = (T.dot(h_, self.U_i))
        preact = wx_preact + uhPrev_preact + self.b_i + user_preact
        if self.useLN: # apply layer normalization to pre-activations
            preact = self.ln(preact, self.b_ln, self.s_ln)
        h = T.tanh(preact)
        if self.drop_p > 0.0: # apply drop-out to post-activations (outer fun)
            h = (h * d_)/(1.0-self.drop_p)
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
                    #p.set_value(numpy.asarray(vals[p.name],dtype=theano.config.floatX))
                else:
                    print('No parameter {} given: default initialization used'.format(p.name))
                    unknown = set(vals.keys()) - {p.name for p in self.params}
                    if len(unknown):
                        print('Unknown parameters {} given'.format(unknown))
