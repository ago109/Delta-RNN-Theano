# -*- coding: utf-8 -*-
#import _pickle as cPickle # Python 3 changed the name
#import _pickle as pkl
from __future__ import unicode_literals
from __future__ import print_function
import csv
import re
import sys, getopt, optparse
import cPickle as pkl
import time
import random
import numpy
import theano
from theano import config
from theano import pp # useful for printing out gradients wrt variables
import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy
import Adam as UpdateRule # <-- import Adam adaptive update rule
import Utils as Util
import LogisticRegression as MaxEnt
import DeltaRNNLayer as DeltaRNN
import LSTMLayer as LSTM
import GRULayer as GRU
import RNNLayer as RNN

from fuel.datasets import TextFile
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Padding, Merge

from collections import OrderedDict
import Utils

# A generic trainer for fitting various RNN-LMs to text data
# All models have been modified to form the user-conditioned generative models
# of the paper.

START_TOKEN = '<start>'
END_TOKEN = '<end>'
SOURCE_FILE = '/home/ago109/IdeaProjects/Delta-RNN-Theano/text.txt'
DICT_FILE = '/home/ago109/IdeaProjects/Delta-RNN-Theano/text_dict.pkl'
ID_DICT_FILE = '/home/ago109/IdeaProjects/Delta-RNN-Theano/id_dict.pkl'

def arrayToStr(arr):
    out = ""
    for i in range(0,len(arr)):
        out += arr[i]
    return out

def sample(a, tau=1.0):
    # helper function to sample an index from a probability array
    a = numpy.log(a) / tau
    a = a - numpy.max(a)
    a = numpy.exp(a) / numpy.sum(numpy.exp(a))
    # https://github.com/llSourcell/How-to-Generate-Music-Demo/issues/4
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.choice.html
    # http://stackoverflow.com/questions/42593231/is-numpy-random-choice-with-replacement-equivalent-to-multinomial-sampling-for-a
    # NOTE: using "choice" avoids pvals sum to 1 bug of numpy.random.multinomial
    return numpy.random.choice(len(a), replace=True, p=a)

def load_id_map(fname):
    idmap = {} # empty dictionary
    numLine = 0
    with open(fname) as f:
        for line in f:
            tok = line.split(",")
            str_id = tok[0]
            num_id = int(tok[1])
            idmap[str_id] = num_id
            numLine += 1
    print("{0} lines read from {1}".format(numLine,fname))
    return idmap


def synthesize(seed = 12345,model='rnn',n_h=50, recThetaName='',outThetaName='',modelPath='',
                temperature=1.0, wght_sd=0.005,out_wght_sd=0.005,useLN=False):

    # Load the datasets with Fuel
    dictionary = pkl.load(open(DICT_FILE, 'r'))
    #dictionary['~'] = len(dictionary)
    reverse_mapping = dict((j, i) for i, j in dictionary.items())

    # Load the datasets with Fuel
    id_dictionary = load_id_map(fname=ID_DICT_FILE)

    n_in = len(dictionary) # number inputs determined by size of lexicon
    print(" > Input.dim = ",n_in)
    n_u = len(id_dictionary)
    print(" > Number.Users = ",n_u)

    print(' > Re-Building model : ',model)

    # Set the random number generator' seeds for consistency
    numpy.random.seed(seed)
    random.seed(seed)
    rng = numpy.random.RandomState(seed)

    x = T.lmatrix('x')
    mask = T.matrix('mask')
    user = T.lmatrix('user')
    dmask = T.tensor3('dmask') # drop-out mask

    # Construct the LSTM layer
    if model == 'lstm':
        recurrent_layer = LSTM.LSTMLayer(rng=rng, input=x, mask=mask, user=user, n_in=n_in, n_h=n_h, n_u=n_u, sd=wght_sd)
    elif model == 'drnn':
        recurrent_layer = DeltaRNN.DeltaRNNLayer(rng=rng, input=x, mask=mask, dmask=dmask, user=user, n_in=n_in, n_h=n_h, n_u=n_u,
                                                 sd=wght_sd, useLN=useLN)
    elif model == 'gru':
        recurrent_layer = GRU.GRULayer(rng=rng, input=x, mask=mask, dmask=dmask, user=user, n_in=n_in, n_h=n_h, n_u=n_u, sd=wght_sd, useLN=useLN)
    elif model == 'rnn':
        recurrent_layer = RNN.RNNLayer(rng=rng, input=x, mask=mask, user=user, dmask=dmask, n_in=n_in, n_h=n_h, n_u=n_u,
                                       sd=wght_sd, useLN=useLN)
    else:
        raise Exception(" Model not understood: ",model)
    recThetaName = "{0}{1}".format(modelPath,recThetaName)
    print(" >> Loading old params for recurrent-layer: ",recThetaName)
    recurrent_layer.load(filename=recThetaName)

    logreg_layer = MaxEnt.LogisticRegression(input=recurrent_layer.output[:-1],
                                      n_in=n_h, n_out=n_in, sd=out_wght_sd)
    outThetaName = "{0}{1}".format(modelPath,outThetaName)
    print(" >> Loading old params for output-layer: ",outThetaName)
    logreg_layer.load(filename=outThetaName)

    batch_size = 1.
    cost = Util.sequence_categorical_crossentropy(logreg_layer.p_y_given_x,
                                             x[1:],
                                             mask[1:]) / batch_size

    # create a list of all model parameters to be fit by gradient descent
    params = logreg_layer.params + recurrent_layer.params

    # Define and compile a function for generating a sequence step by step.
    x_t = T.iscalar()
    h_p = T.matrix()
    u_t = T.iscalar()
    if model == 'lstm': # LSTMs have separate memory cell c
        c_p = T.matrix()
        h_t, c_t = recurrent_layer._step(T.ones(1), x_t, u_t, T.ones(1), h_p, c_p)
        energy = T.dot(h_t, logreg_layer.W) + logreg_layer.b

        energy_exp = T.exp(energy - T.max(energy, 1)[:, None])

        output = energy_exp / energy_exp.sum(1)[:, None]
        single_step = theano.function([x_t, u_t, h_p, c_p], [output, h_t, c_t])
    else: # any other of the models tested in this paper only maintain 1 state
        h_t = recurrent_layer._step(T.ones(1),x_t, u_t, T.ones(1), h_p)
        energy = T.dot(h_t, logreg_layer.W) + logreg_layer.b

        energy_exp = T.exp(energy - T.max(energy, 1)[:, None])

        output = energy_exp / energy_exp.sum(1)[:, None]
        single_step = theano.function([x_t, u_t, h_p], [output, h_t])

    numParams = recurrent_layer.getNumParams() + logreg_layer.getNumParams()
    print(" -> Number Parameters = ",numParams)

    start_time = time.clock()

    writer = csv.writer(open(modelPath + "synthesized_tweets.csv", 'w'),quoting=csv.QUOTE_NONNUMERIC)
    #writer = csv.writer(open(modelPath + "synthesized_tweets.csv", 'wb'),quoting=csv.QUOTE_NONNUMERIC)
    writer.writerow(['id','user_id','text'])

    # Synthesize some text from the generative model at this stage/epoch
    print(" > Synthesizing data from model...")
    dummy_val = [numpy.ones(1).astype(theano.config.floatX)]
    with open(SOURCE_FILE) as f:
        line_num = 0
        for line in f:
            if line_num > 0:
                tok = re.split(",",line)
                l_id = tok[0].strip(' ').strip("\"")
                u_id = tok[1].strip(' ').strip("\"") # original key string-name
                prediction = numpy.ones(n_in, dtype=config.floatX) / (n_in * 1.0)
                h_p = numpy.zeros((n_h,), dtype=config.floatX)
                #sentence = START_TOKEN
                sentence = "" # we omit start-tokens as a post-processing step
                x_t = dictionary[START_TOKEN]
                u_t = id_dictionary[u_id]
                if model == 'lstm':
                    c_p = numpy.zeros((n_h,), dtype=config.floatX)
                    prediction, h_p, c_p = single_step(x_t, u_t, [h_p.flatten()],[c_p.flatten()])
                else:
                    prediction, h_p = single_step(x_t, u_t, [h_p.flatten()])
                #s_t = numpy.random.multinomial(1, prediction.flatten())
                s_t = sample(prediction.flatten(), tau=temperature)
                t = 0
                stayInLoop = True
                while(t < 200 and stayInLoop): # Tweet maximum is 200 characters!
                    x_t = s_t # numpy.argmax(s_t)
                    if model == 'lstm':
                        prediction, h_p, c_p = single_step(x_t, u_t, [h_p.flatten()],[c_p.flatten()])
                    else:
                        prediction, h_p = single_step(x_t, u_t, [h_p.flatten()])
                    symbol = reverse_mapping[x_t]
                    if symbol == "_": # simple post-processing to convert spaces back
                        symbol = " "
                    if symbol == "<us>": # simple post-processing to convert spaces back
                        symbol = "_"
                    if symbol == END_TOKEN: # detect an end-token
                        stayInLoop = False
                    else:
                        sentence += symbol
                    #s_t = numpy.random.multinomial(1, prediction.flatten())
                    s_t = sample(prediction.flatten(), tau=temperature)
                    t = t + 1
                #print('MODEL({0},{1}): {2}'.format(l_id,u_id,sentence))
                #out_line = "\"{0}\", \"{1}\", \"{2}\"".format(l_id, u_id, sentence)
                writer.writerow(["{0}".format(l_id),"{0}".format(u_id),"{0}".format(sentence)])
                print("\r  {0} samples generated...".format(line_num),end='')
            line_num = line_num + 1
        print("")


# Main method for Python (execute code here)
if __name__ == '__main__':
    # Execute main program (since it will most likely be from the command line)
    seed = 12
    sourcefile = '' # data source to build dictionary from
    id_dictfname = ''
    model = 'rnn'
    n_h = 20
    recThetaName = ''
    outThetaName = ''
    modelPath = ''
    temperature = 1.0
    useLN = False

    options, remainder = getopt.getopt(sys.argv[1:], '', ["seed=","sourcefile=","dictfname=",
                        "id_dictfname=","model=","n_h=","recThetaName=","outThetaName=","modelPath=",
                        "temperature=","useLN="])

    # Collect arguments ifrom argv
    for opt, arg in options:
        if opt in ("--sourcefile"):
            sourcefile = arg
            SOURCE_FILE = sourcefile
        elif opt in ("--dictfname"):
            dictfname = arg
            DICT_FILE = dictfname
        elif opt in ("--id_dictfname"):
            id_dictfname = arg
            ID_DICT_FILE = id_dictfname
        elif opt in ("--model"):
            model = arg
        elif opt in ("--n_h"):
            n_h = int(arg)
        elif opt in ("--temperature"):
            temperature = float(arg)
        elif opt in ("--seed"):
            seed = int(arg)
        elif opt in ("--useLN"):
            truth = int(arg)
            if truth == 1:
                useLN = True
            # else, default is False
        elif opt in ("--recThetaName"):
            recThetaName = arg
        elif opt in ("--outThetaName"):
            outThetaName = arg
        elif opt in ("--modelPath"):
            modelPath = arg

    synthesize(seed=seed,model=model,n_h=n_h, recThetaName=recThetaName,outThetaName=outThetaName,modelPath=modelPath,
               temperature=temperature,useLN=useLN)
