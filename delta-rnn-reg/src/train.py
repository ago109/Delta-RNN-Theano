# -*- coding: utf-8 -*-
#import _pickle as cPickle # Python 3 changed the name
#import _pickle as pkl
from __future__ import print_function
import csv
import sys, getopt, optparse
import cPickle as pkl
import time
import random
import numpy
import theano
import math
from theano import config
from theano import pp # useful for printing out gradients wrt variables
import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy
import UpdateRule # import Adam as UpdateRule # <-- import Adam adaptive update rule
import Utils as Util
import LogisticRegression as MaxEnt
import DeltaRNNLayer as DeltaRNN
import GRULayer as GRU

from fuel.datasets import TextFile
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Padding, Merge

from collections import OrderedDict

import Utils

# A generic trainer for fitting various RNN-LMs to text data
# All models have been modified to form the user-conditioned generative models
# of the paper.

START_TOKEN = '<s>'
END_TOKEN = '</s>'
TRAIN_FILE = ''
VAL_FILE = ''
DICT_FILE = ''
useFanInFanOut = 0
useOrtho = 0
l_r_decay = 0.5
drop_inner = 0.0

def arrayToStr(arr):
    out = ""
    for i in range(0,len(arr)):
        out += arr[i]
    return out

def train_model(seed = 12345,model='rnn',batch_size=50, n_h=50, n_epochs=40, updater='sgd', lr = 0.002,
    recThetaName='',outThetaName='',modelPath='',error_mark=40,fit_model=1,wght_sd=0.005,out_wght_sd=0.005,
    useLN=False, drop_p=0., grad_clip=0, patience=1, dyn_eval=0, funtype="identity", norm_max=-1.0):
	
    # load in feature dictionary
    dictionary = pkl.load(open(DICT_FILE, 'r'))
    #dictionary['~'] = len(dictionary)
    reverse_mapping = dict((j, i) for i, j in dictionary.items())

    n_in = len(dictionary) # number inputs determined by size of lexicon
    print(" > Input.dim = ",n_in)

    # Load the datasets with Fuel
    print(" > Loading train data: ",TRAIN_FILE)
    train = TextFile(files=[TRAIN_FILE],
                     dictionary=dictionary,
                     unk_token=None,
                     level='word',
                     preprocess=None,
                     bos_token=None,
                     eos_token=None)

    train_stream = DataStream.default_stream(train) # get text-stream

    # organize data in batches and pad shorter sequences with zeros
    train_stream = Batch(train_stream,
                         iteration_scheme=ConstantScheme(batch_size))
    train_stream = Padding(train_stream)

    # idem dito for the validation text
    print(" > Loading valid data: ",VAL_FILE)
    val = TextFile(files=[VAL_FILE],
                     dictionary=dictionary,
                     unk_token=None,
                     level='word',
                     preprocess=None,
                     bos_token=None,
                     eos_token=None)

    val_stream = DataStream.default_stream(val)

    # organize data in batches and pad shorter sequences with zeros
    val_stream = Batch(val_stream,
                         iteration_scheme=ConstantScheme(batch_size))

    # pad text-token sequences & user-id sequences
    val_stream = Padding(val_stream)

    print(' > Building model : ',model)

    # Set the random number generator' seeds for consistency
    numpy.random.seed(seed)
    random.seed(seed)
    rng = numpy.random.RandomState(seed)

    x = T.lmatrix('x')
    mask = T.matrix('mask')
    dmask = T.tensor3('dmask') # drop-out mask

    # Construct the LSTM layer
    if model == 'drnn':
        recurrent_layer = DeltaRNN.DeltaRNNLayer(rng=rng, input=x, mask=mask, dmask=dmask, n_in=n_in, n_h=n_h,
                                                 sd=wght_sd,useLN=useLN,drop_p=drop_p,funtype=funtype,useFanInFanOut=useFanInFanOut,useOrtho=useOrtho,drop_inner=drop_inner)
    elif model == 'gru':
        recurrent_layer = GRU.GRULayer(rng=rng, input=x, mask=mask, dmask=dmask, n_in=n_in, n_h=n_h, sd=wght_sd,useLN=useLN,drop_p=drop_p,useFanInFanOut=useFanInFanOut,useOrtho=useOrtho)
    else:
        raise Exception(" Model not understood: ",model)
    if len(recThetaName) > 0: # Load in any pre-built parameters for recurrent layer
        print(" >> Loading old params for recurrent-layer: ",recThetaName)
        recurrent_layer.load(filename=recThetaName)

    print(" Using fan-in-fan-out for softmax weights? {0}".format(useFanInFanOut))
    logreg_layer = MaxEnt.LogisticRegression(input=recurrent_layer.output[:-1],
                                      n_in=n_h, n_out=n_in, sd=out_wght_sd, useFanInFanOut=useFanInFanOut, useOrtho=useOrtho)
    if len(outThetaName) > 0: # Load in any pre-built parameters for output layer
        print(" >> Loading old params for output-layer: ",outThetaName)
        logreg_layer.load(filename=outThetaName)

    cost = Util.sequence_categorical_crossentropy(logreg_layer.p_y_given_x,
                                             x[1:],
                                             mask[1:]) / batch_size

    # create a list of all model parameters to be fit by gradient descent
    params = logreg_layer.params + recurrent_layer.params

    # create a list of gradients for all model parameters
    if grad_clip > 0.0:
        print(" >> Clipping grads of magnitude > ",grad_clip)
        #grads = T.grad(cost, params, disconnected_inputs='ignore')
        #grad_lst = [ T.sum( (  grad / float(batch_size) )**2  ) for grad in grads ]
        #grad_norm = T.sqrt( T.sum( grad_lst ))
        #all_grads = ifelse(T.gt(grad_norm, max_norm),
        #           [grads*(max_norm / grad_norm) for grads in all_grads],
        #           all_grads)
        grads = T.grad(theano.gradient.grad_clip(cost, -1 * grad_clip, grad_clip), params, disconnected_inputs='ignore')
    else:
        grads = T.grad(cost, params, disconnected_inputs='ignore')

    # set up update rules for model parameters
    print(" Clipping param norms to max of {0}".format(norm_max))
    #learning_rate = lr
    learning_rate = T.scalar('learning_rate', dtype=theano.config.floatX)
    if updater == 'sgd': # use classical SGD
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
        ]
    elif updater == 'adam': # use Adam adaptive learning rate update rule
        grads = OrderedDict(zip(params, grads))
        updates = UpdateRule.Adam(grads, learning_rate, norm_max=norm_max)
    elif updater == 'rmsprop': # use RMSprop adaptive learning rate update rule
        grads = OrderedDict(zip(params, grads))
        updates = UpdateRule.RMSprop(grads, learning_rate, norm_max=norm_max)
    else:
        raise Exception("Updater not understood: ",updater)

    update_model = theano.function([x, mask, dmask, learning_rate], cost, updates=updates) #, allow_input_downcast=True)

    evaluate_model = theano.function([x, mask, dmask], cost) #, allow_input_downcast=True)

    numParams = recurrent_layer.getNumParams() + logreg_layer.getNumParams()
    print(" -> Number Parameters = ",numParams)

    start_time = time.clock()

    iteration = 0
    print(" Random.NLL = ",numpy.log(1. * n_in))
    best_nll = -1.0

    if fit_model == 1: # FIT MODEL TO THE DATA IF FLAG RAISED

        logfd = open(modelPath + "performance.csv", 'wb')
        writer = csv.writer(logfd)
        writer.writerow(["Epoch","AVG_NLL","BPC","AVG_PPL"])

        # Get initial scores before any training
        val_scores = []
        N = 0
        nll = 0.

        for x_val, x_mask in val_stream.get_epoch_iterator():
            # 3D tensor shape:  variable dim x batch-size dim x time-window dim
            #d_mask = Utils.create_ones(n_h,x_mask.shape[0],x_mask.shape[1])
            d_mask = Utils.create_zone_out_mask(rng, n_h, x_mask.shape[0],x_mask.shape[1], drop_p, sample=False)
            batch_score = evaluate_model(x_val.T, x_mask.T, d_mask.T)
            #batch_score = 0.0
            nll += batch_score * x_mask.shape[0]
            val_scores.append(batch_score)
            N += numpy.sum(x_mask)
        nll = nll / N
        ce = numpy.mean(val_scores)
        ppl = numpy.exp(nll)
        print(' >> Epoch = {0} NLL = {1} log2(NLL) = {2} exp(NLL) = {3} '.format(-1,nll,(nll/math.log(2)),ppl))
        writer.writerow([0,nll,(nll/math.log(2)),ppl])
        logfd.flush()
        best_nll = nll
        best_epoch = -1

        impatience = 0
        l_r = lr # set initial learning rate
        for epoch in range(n_epochs):
            print('Epoch:', epoch)

            improve_flag = False
            for x_, x_mask_ in train_stream.get_epoch_iterator():
                iteration += 1
                #print("\r  {0} mini-batches seen...".format(iteration),end='')
                #d_mask = Utils.create_ones(n_h,x_mask_.shape[0],x_mask_.shape[1])
                d_mask = Utils.create_zone_out_mask(rng, n_h, x_mask_.shape[0],x_mask_.shape[1], drop_p)
                cross_entropy = update_model(x_.T, x_mask_.T, d_mask.T, l_r)
                #print("\r  {0} mini-batches seen CE = {1}".format(iteration,cross_entropy),end='')
                #print("  {0} --> {1} mini-batches seen CE = {2}".format(epoch,iteration,cross_entropy))

                if iteration % error_mark == 0:
                    #print("")
                    #print('epoch:', epoch, '  minibatch:', iteration)
                    val_scores = []
                    N = 0
                    nll = 0.
                    for x_val, x_mask in val_stream.get_epoch_iterator():
			if x_val.size > 0: # as long as sample sequence is non-emtpy
	                        #d_mask = Utils.create_ones(n_h,x_mask.shape[0],x_mask.shape[1])
	                        d_mask = Utils.create_zone_out_mask(rng, n_h, x_mask.shape[0],x_mask.shape[1], drop_p, sample=False)
	                        batch_score = evaluate_model(x_val.T, x_mask.T, d_mask.T)
	                        nll += batch_score * x_mask.shape[0] # un-normalize mini-batch scores
	                        val_scores.append(batch_score)
	                        N += numpy.sum(x_mask)
                    nll = nll / N
                    ce = numpy.mean(val_scores)
                    ppl = numpy.exp(nll)
                    writer.writerow([(epoch),nll,(nll/math.log(2)),ppl])
                    logfd.flush()
                    if nll < best_nll:
                        best_nll = nll
                        best_epoch = epoch
                        print(" >> Saving best model at epoch {0} with NLL = {1}".format(best_epoch, best_nll))
                        # Check-point save at end of epoch
                        recSave = "{0}rec-params-best-{1}".format(modelPath,epoch)
                        recurrent_layer.save(recSave)
                        outSave = "{0}out-params-best-{1}".format(modelPath,epoch)
                        logreg_layer.save(outSave)
                        # Save best params so far
                        recSave = "{0}rec-params-best".format(modelPath,epoch)
                        recurrent_layer.save(recSave)
                        outSave = "{0}out-params-best".format(modelPath,epoch)
                        logreg_layer.save(outSave)
                        improve_flag = True # raise improvement flag since improvement was observed
                    print(' >> Epoch = {0} Avg.NLL = {1} (Best = {5}) Avg.BPC = {2} PPL = {3} Iter = {4}'.format(epoch,nll,(nll/math.log(2)),ppl,iteration,best_nll))

                    # adapt the learning rate based on patience schedule
                    if improve_flag is False:
                        if patience > 0: # we only consider positive/non-zero patience values (otherwise, turn this option off)
                            impatience += 1
                            if impatience >= patience:
                                l_r = (numpy.maximum(1e-4,l_r * l_r_decay)).astype(theano.config.floatX)
                                print(" __Decreasing learning rate to ",l_r)
                                impatience = 0
                    improve_flag = False


            # Evaluate generalization at end of epoch
            if iteration % error_mark != 0: # this if-stmt avoids redundant evaluation computation
                #print("")
                val_scores = []
                N = 0
                nll = 0.
                for x_val, x_mask in val_stream.get_epoch_iterator():
			if x_val.size > 0: # as long as sample sequence is non-emtpy
	                    #d_mask = Utils.create_ones(n_h,x_mask.shape[0],x_mask.shape[1])
	                    d_mask = Utils.create_zone_out_mask(rng, n_h, x_mask.shape[0],x_mask.shape[1], drop_p, sample=False)
	                    batch_score = evaluate_model(x_val.T, x_mask.T, d_mask.T)
        	            nll += batch_score * x_mask.shape[0]
	                    val_scores.append(batch_score)
	                    N += numpy.sum(x_mask)
                nll = nll / N
                ce = numpy.mean(val_scores)
                ppl = numpy.exp(nll)

                writer.writerow([(epoch+1),nll,(nll/math.log(2)),ppl])
                logfd.flush()
                # Check-point
                recSave = "{0}rec-params-end-{1}".format(modelPath,epoch)
                recurrent_layer.save(recSave)
                outSave = "{0}out-params-end-{1}".format(modelPath,epoch)
                logreg_layer.save(outSave)
                if nll < best_nll:
                    best_nll = nll
                    best_epoch = epoch
                    print(" >> Saving best model at epoch {0} with NLL = {1}".format(best_epoch, best_nll))
                    # Check-point save at end of epoch
                    recSave = "{0}rec-params-best-{1}".format(modelPath,epoch)
                    recurrent_layer.save(recSave)
                    outSave = "{0}out-params-best-{1}".format(modelPath,epoch)
                    logreg_layer.save(outSave)
                    # Save best params so far
                    recSave = "{0}rec-params-best".format(modelPath,epoch)
                    recurrent_layer.save(recSave)
                    outSave = "{0}out-params-best".format(modelPath,epoch)
                    logreg_layer.save(outSave)
                    improve_flag = True
                print(' >> Epoch = {0} Avg.NLL = {1} (Best = {4}) Avg.BPC = {2} PPL = {3} '.format(epoch,nll,(nll/math.log(2)),ppl,best_nll))

            # adapt the learning rate based on patience schedule
            '''
            if improve_flag is False:
                if patience > 0:
                    impatience += 1
                    if impatience >= patience:
                        l_r = (numpy.maximum(0.00001,l_r / 2.0)).astype(theano.config.floatX)
                        print(" __Decreasing learning rate to ",l_r)
                        impatience = 0
            '''

        print("")
        print(' > Optimization complete.')
        print(' >>>> Best NLL = {0} at Epoch {1}'.format(best_nll,best_epoch))
        end_time = time.clock()
        print(' > The code ran for %.2fm' % ((end_time - start_time) / 60.))
        print('---------------------------------------')
	logfd.close()
    else:
        print(' > Skipping model fit directly to evaluation...')
        print(' > dynamic eval code := ',dyn_eval)
        # EVALUATION-ONLY
        print(' > FINAL.VALID := ',VAL_FILE)
        val_scores = []
        N = 0
        nll = 0.
        l_r = lr # set initial learning rate
        for x_val, x_mask in val_stream.get_epoch_iterator():
                        #print(' type = ',type(x_val))
                        #x_val = x_val.astype(numpy.int64)
                        #print(x_val)
			#print('  = ',(x_val).size)
                        if x_val.size > 0: # as long as sample sequence is non-emtpy
				if dyn_eval > 0:
					d_mask = Utils.create_zone_out_mask(rng, n_h, x_mask.shape[0],x_mask.shape[1], drop_p)
					batch_score = update_model(x_val.T, x_mask.T, d_mask.T, l_r)
				else:
					#d_mask = Utils.create_ones(n_h,x_mask.shape[0],x_mask.shape[1])
					d_mask = Utils.create_zone_out_mask(rng, n_h, x_mask.shape[0],x_mask.shape[1], drop_p, sample=False)
					batch_score = evaluate_model(x_val.T, x_mask.T, d_mask.T)
				#print(" B{0} vs X{1} N{2}".format(batch_size,x_mask.shape[0],numpy.sum(x_mask)))
				nll += batch_score * batch_size #* x_mask.shape[0] # un-normalize mini-batch scores
				val_scores.append(batch_score)
				N += numpy.sum(x_mask)
				print('\r NLL.tmp = {0} over {1}'.format((nll / N),N),end='')
        print('')
        nll = nll / N
        ce = numpy.mean(val_scores)
        ppl = numpy.exp(nll)
        print(' > FINAL.VALID: Avg.NLL = {0} Avg.BPC = {1} PPL = {2} N.tokens = {3}'.format(nll,(nll/math.log(2)),ppl,N))
		
        if dyn_eval > 1:
        	dyn_eval = 0

        # Evaluate model on training as well (as measure of overfitting)
        print(' > FINAL.TRAIN := ',TRAIN_FILE)
        val_scores = []
        N = 0
        nll = 0.
        for x_val, x_mask in train_stream.get_epoch_iterator():
			if x_val.size > 0: # as long as sample sequence is non-emtpy
				if dyn_eval > 0:
					d_mask = Utils.create_zone_out_mask(rng, n_h, x_mask.shape[0],x_mask.shape[1], drop_p)
					batch_score = update_model(x_val.T, x_mask.T, d_mask.T, l_r)
				else:
					#d_mask = Utils.create_ones(n_h,x_mask.shape[0],x_mask.shape[1])
					d_mask = Utils.create_zone_out_mask(rng, n_h, x_mask.shape[0],x_mask.shape[1], drop_p, sample=False)
					batch_score = evaluate_model(x_val.T, x_mask.T, d_mask.T)
				nll += batch_score * batch_size # * x_mask.shape[0] # un-normalize mini-batch scores
				val_scores.append(batch_score)
				N += numpy.sum(x_mask)
				print('\r NLL.tmp = {0} over {1}'.format((nll / N),N),end='')
        print('')
        nll = nll / N
        ce = numpy.mean(val_scores)
        ppl = numpy.exp(nll)
        print(' > FINAL.TRAIN: Avg.NLL = {0} Avg.BPC = {1} PPL = {2} N.tokens = {3}'.format(nll,(nll/math.log(2)),ppl,N))

        end_time = time.clock()
        print(' > Final Evaluation complete.')
        print(' > The code ran for %.2fm' % ((end_time - start_time) / 60.))
        print('---------------------------------------')
        recSave = modelPath + "rec-params"
        print(' > Saving model.recurrent params to disk: ',recSave)
        recurrent_layer.save(recSave)
        outSave = modelPath + "out-params"
        print(' > Saving model.output params to disk: ',outSave)
        logreg_layer.save(outSave)

# Main method for Python (execute code here)
if __name__ == '__main__':
    # Execute main program (since it will most likely be from the command line)
    seed = 12
    trainfile = '' # data source to build dictionary from
    trainIDfile = ''
    dictfname = '' # name of output lexicion
    id_dictfname = ''
    validfile = '' # default delim is space character
    validIDfile = ''
    model = 'rnn'
    batch_size = 1
    n_h = 20
    n_epochs = 50
    updater = 'sgd'
    lr = 0.01
    recThetaName = ''
    outThetaName = ''
    modelPath = ''
    error_mark = 40
    fit_model = 1
    wght_sd= 0.005
    out_wght_sd = 0.005
    useLN = False
    drop_p = 0.
    grad_clip = 0.
    patience = 2
    dyn_eval = 0
    funtype = "identity"
    norm_max = -1.0
    drop_inner = 0.0

    options, remainder = getopt.getopt(sys.argv[1:], '', ["seed=","trainfile=","dictfname=",
                        "validfile=","model=","batch_size=","n_h=","n_epochs=",
                        "updater=","lr=","wght_sd=","out_wght_sd=","updater=","recThetaName=","outThetaName=",
                        "modelPath=","error_mark=","fit_model=","useLN=","drop_p=","grad_clip=","patience=","dyn_eval=","funtype=","norm_max=","useFanInFanOut=","lr_decay=","useOrtho=","drop_inner="])

    # Collect arguments ifrom argv
    for opt, arg in options:
        if opt in ("--trainfile"):
            trainfile = arg
            TRAIN_FILE = trainfile
        elif opt in ("--dictfname"):
            dictfname = arg
            DICT_FILE = dictfname
        elif opt in ("--validfile"):
            validfile = arg
            VAL_FILE = validfile
        elif opt in ("--model"):
            model = arg
        elif opt in ("--useLN"):
            truth = int(arg)
            if truth == 1:
                useLN = True
            # else, default is False
        elif opt in ("--n_h"):
            n_h = int(arg)
        elif opt in ("--error_mark"):
            error_mark = int(arg)
        elif opt in ("--n_epochs"):
            n_epochs = int(arg)
        elif opt in ("--seed"):
            seed = int(arg)
        elif opt in ("--patience"):
            patience = int(arg)
        elif opt in ("--lr"):
            lr = float(arg)
        elif opt in ("--grad_clip"):
            grad_clip = float(arg)
        elif opt in ("--drop_p"):
            drop_p = float(arg)
        elif opt in ("--wght_sd"):
            wght_sd = float(arg)
        elif opt in ("--out_wght_sd"):
            out_wght_sd = float(arg)
        elif opt in ("--batch_size"):
            batch_size = int(arg)
        elif opt in ("--updater"):
            updater = arg
        elif opt in ("--recThetaName"):
            recThetaName = arg
        elif opt in ("--outThetaName"):
            outThetaName = arg
        elif opt in ("--modelPath"):
            modelPath = arg
        elif opt in ("--fit_model"):
            fit_model = int(arg)
        elif opt in ("--dyn_eval"):
            dyn_eval = int(arg)
        elif opt in ("--funtype"):
            funtype = arg
        elif opt in ("--norm_max"):
            norm_max = float(arg)
        elif opt in ("--useFanInFanOut"):
            useFanInFanOut = int(arg)
        elif opt in ("--useOrtho"):
            useOrtho = int(arg)
        elif opt in ("--lr_decay"):
            l_r_decay = float(arg)
        elif opt in ("--drop_inner"):
            drop_inner = float(arg)

    train_model(seed=seed,model=model,batch_size=batch_size, n_h=n_h, n_epochs=n_epochs,
                updater=updater, lr=lr, recThetaName=recThetaName,outThetaName=outThetaName,
                modelPath=modelPath,error_mark=error_mark,fit_model=fit_model,wght_sd=wght_sd,
                out_wght_sd=out_wght_sd,useLN=useLN,drop_p=drop_p,grad_clip=grad_clip,patience=patience,dyn_eval=dyn_eval,funtype=funtype,norm_max=norm_max)

