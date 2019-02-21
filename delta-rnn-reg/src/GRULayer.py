import logging
import numpy
import theano
from theano import config
import theano.tensor as T
import Utils as Util

# Construct a GRU
class GRULayer:
    '''
    Gated Recurrent Unit (GRU) baseline for (Ororbia et al., 2017) - Learning Simpler Language Models with the Differential State Framework


    Custom implementation of GRU, which can employ drop-out, layer normalization and sports
    either gaussian, fan-in-fan-out (xavier), or random orthogonal initialization schemes.
    
    @author - Alexander G. Ororbia II
    '''

    def __init__(self, rng, input, mask, dmask, n_in, n_h, sd=.005, useLN=False, drop_p=0.0, useFanInFanOut=0, useOrtho=0):
        # Init params
        self.drop_p = drop_p
        self.useLN = useLN

	if useFanInFanOut == 1:
            print(" Using fan-in-fan-out init for recurrent params!")
            self.W_r = theano.shared(Util.fanInFanOut((n_in, n_h)), 'W_r', borrow=True)
            self.numParams = n_in * n_h
            self.U_r = theano.shared(Util.fanInFanOut((n_h, n_h)), 'U_r', borrow=True)
            self.numParams += n_h * n_h
            self.W_z = theano.shared(Util.fanInFanOut((n_in, n_h)), 'W_z', borrow=True)
            self.numParams += n_in * n_h
            self.U_z = theano.shared(Util.fanInFanOut((n_h, n_h)), 'U_z', borrow=True)
            self.numParams += n_h * n_h
            self.W_h = theano.shared(Util.fanInFanOut((n_in, n_h)), 'W_h', borrow=True)
            self.numParams += n_in * n_h
            self.U_h = theano.shared(Util.fanInFanOut((n_h, n_h)), 'U_h', borrow=True)
            self.numParams += n_h * n_h
        elif useOrtho == 1:
            print(" Using orthogonal init for recurrent params!")
            self.W_r = theano.shared(Util.rand_ortho_u((n_in, n_h)), 'W_r', borrow=True)
            self.numParams = n_in * n_h
            self.U_r = theano.shared(Util.rand_ortho_u((n_h, n_h)), 'U_r', borrow=True)
            self.numParams += n_h * n_h
            self.W_z = theano.shared(Util.rand_ortho_u((n_in, n_h)), 'W_z', borrow=True)
            self.numParams += n_in * n_h
            self.U_z = theano.shared(Util.rand_ortho_u((n_h, n_h)), 'U_z', borrow=True)
            self.numParams += n_h * n_h
            self.W_h = theano.shared(Util.rand_ortho_u((n_in, n_h)), 'W_h', borrow=True)
            self.numParams += n_in * n_h
            self.U_h = theano.shared(Util.rand_ortho_u((n_h, n_h)), 'U_h', borrow=True)
            self.numParams += n_h * n_h
        else:      
            self.W_r = theano.shared(Util.gauss_weight(n_in, n_h, sd=sd), 'W_r', borrow=True)
            self.numParams = n_in * n_h
            self.U_r = theano.shared(Util.gauss_weight(n_h, sd=sd), 'U_r', borrow=True)
            self.numParams += n_h * n_h
            self.W_z = theano.shared(Util.gauss_weight(n_in, n_h, sd=sd), 'W_z', borrow=True)
            self.numParams += n_in * n_h
            self.U_z = theano.shared(Util.gauss_weight(n_h, sd=sd), 'U_z', borrow=True)
            self.numParams += n_h * n_h
            self.W_h = theano.shared(Util.gauss_weight(n_in, n_h, sd=sd), 'W_h', borrow=True)
            self.numParams += n_in * n_h
            self.U_h = theano.shared(Util.gauss_weight(n_h, sd=sd), 'U_h', borrow=True)
            self.numParams += n_h * n_h 
       

        if useLN: # a GRU has 14 layer normalizations!
            # no need for standard biases, since LN parameters serve this role
            self.bz1_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'bz1_ln', borrow=True)
            self.sz1_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'sz1_ln', borrow=True)
            self.bz2_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'bz2_ln', borrow=True)
            self.sz2_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'sz2_ln', borrow=True)
            self.br1_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'br1_ln', borrow=True)
            self.sr1_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'sr1_ln', borrow=True)
            self.br2_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'br2_ln', borrow=True)
            self.sr2_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'sr2_ln', borrow=True)
            self.bh1_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'bg1_ln', borrow=True)
            self.sh1_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'sh1_ln', borrow=True)
            self.bh2_ln = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'bg2_ln', borrow=True)
            self.sh2_ln = theano.shared(numpy.ones((n_h,), dtype=config.floatX),
                                     'sh2_ln', borrow=True)
            self.numParams += (n_h * 12)
            self.params = [self.W_r, self.U_r, self.W_z, self.U_z,
                           self.W_h, self.U_h, self.bz1_ln, self.sz1_ln,
                           self.bz2_ln, self.sz2_ln, self.br1_ln, self.sr1_ln, self.br2_ln,
                           self.sr2_ln, self.bh1_ln, self.sh1_ln, self.bh2_ln, self.sh2_ln]
        else:
            self.b_r = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'b_r', borrow=True)
            self.numParams += n_h
            self.b_z = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'b_z', borrow=True)
            self.numParams += n_h
            self.b_h = theano.shared(numpy.zeros((n_h,), dtype=config.floatX),
                                     'b_h', borrow=True)
            self.numParams += n_h
            self.params = [self.W_r, self.U_r, self.b_r,
                           self.W_z, self.U_z, self.b_z,
                           self.W_h, self.U_h, self.b_h]

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
        # Layer normalization is applied as described in appendix of: https://arxiv.org/pdf/1607.06450.pdf
        if self.useLN:
            rp1 = self.ln(Util.index_dot(x_, self.W_r), self.br1_ln, self.sr1_ln)
            rp2 = self.ln(T.dot(h_, self.U_r), self.br2_ln, self.sr2_ln)
            r_preact = rp1 + rp2
        else:
            r_preact = Util.index_dot(x_, self.W_r) + T.dot(h_, self.U_r) + self.b_r
        r = T.nnet.sigmoid(r_preact)

        if self.useLN:
            zp1 = self.ln(Util.index_dot(x_, self.W_z), self.bz1_ln, self.sz1_ln)
            zp2 = self.ln(T.dot(h_, self.U_z), self.bz2_ln, self.sz2_ln)
            z_preact = zp1 + zp2
        else:
            z_preact = Util.index_dot(x_, self.W_z) + T.dot(h_, self.U_z) + self.b_z
        z = T.nnet.sigmoid(z_preact)

        if self.useLN:
            gp1 = self.ln(Util.index_dot(x_, self.W_h), self.bh1_ln, self.sh1_ln)
            gp2 = self.ln(T.dot(r * h_, self.U_h), self.bh2_ln, self.sh2_ln)
            g_preact = gp1 + (gp2)
        else:
            g_preact = Util.index_dot(x_, self.W_h) + T.dot(r * h_, self.U_h) + self.b_h
        g = T.tanh(g_preact) #hHat

        h = z * g + (1. - z) * h_
        if self.drop_p > 0.0: # apply drop-out to post-activations (outer fun)
            h = (h * d_) #/(1.0-self.drop_p) # h = (h /(1.0-self.drop_p)) * d_
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
