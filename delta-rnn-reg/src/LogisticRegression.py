import logging
import cPickle as pkl
import time

import numpy
import theano
from theano import config
import theano.tensor as T
import Utils as Util

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    Logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, sd=.005, useFanInFanOut=0, useOrtho=0):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie 

        """

        # initialize the weights W as a matrix of shape (n_in, n_out)
        if useFanInFanOut == 1:
            print(" Using fan-in-fan-out for output param init!")
            self.W = theano.shared(value=Util.fanInFanOut((n_in, n_out)), name='W', borrow=True)
        elif useOrtho == 1:
            print(" Using random orthogonal for output param init!")
            self.W = theano.shared(value=Util.rand_ortho_u((n_in, n_out)), name='W', borrow=True)
        else:
            self.W = theano.shared(value=Util.gauss_weight(n_in, n_out, sd = sd), name='W', borrow=True)
        self.numParams = n_in * n_out
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        self.numParams += n_out

        # compute vector of class-membership probabilities in symbolic form
        energy = T.dot(input, self.W) + self.b
        energy_exp = T.exp(energy - T.max(energy, 2)[:, :, None])
        pmf = energy_exp / energy_exp.sum(2)[:, :, None]
        self.p_y_given_x = pmf

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

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
