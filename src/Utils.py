import numpy
import theano
from theano import config
import theano.tensor as T
from theano.tensor.nnet import categorical_crossentropy

def sequence_categorical_crossentropy(prediction, targets, mask):
    prediction_flat = prediction.reshape(((prediction.shape[0] * prediction.shape[1]), prediction.shape[2]), ndim=2)
    targets_flat = targets.flatten()
    mask_flat = mask.flatten()
    ce = categorical_crossentropy(prediction_flat, targets_flat)
    return T.sum(ce * mask_flat)

def fanInFanOut(shape):
    nIn = shape[0]
    nOut = shape[1]
    irange = numpy.sqrt(6./(nIn + nOut))
    return rand(shape, irange)

def rand(shape, irange):
    return numpy.asarray( - irange + 2 * irange * numpy.random.rand(*shape), dtype=config.floatX)

def rand_ortho_u(shape):
    nIn = shape[0]
    nOut = shape[1]
    irange = numpy.sqrt(6./(nIn + nOut))
    return rand_ortho(shape, irange)

def rand_ortho(shape, irange) : 
    A = - irange + 2 * irange * numpy.random.rand(*shape)
    U, s, V = numpy.linalg.svd(A, full_matrices=True)
    return numpy.asarray(  numpy.dot(U, numpy.dot( numpy.eye(U.shape[1], V.shape[0]), V )), dtype=config.floatX)

def gauss_weight(ndim_in, ndim_out=None, sd=.005):
    if ndim_out is None:
        ndim_out = ndim_in
    W = numpy.random.randn(ndim_in, ndim_out) * sd
    return numpy.asarray(W, dtype=config.floatX)

def zero_weight(ndim_in, ndim_out=None):
    if ndim_out is None:
        ndim_out = ndim_in
    return numpy.zeros((ndim_in,ndim_out), dtype=config.floatX)

def index_dot(indices, w):
    return w[indices.flatten()]

def create_ones(nx, ny, nz):
    return numpy.ones((nx,ny,nz), dtype=config.floatX)

def create_drop_out_mask(rng, n_x, n_s, n_t, p, sample=True): # 3D tensor shape:  variable dim x batch-size dim x time-window dim
    if sample and p > 0.0: # compute drop-out masks (i.e., draw sampless
        time = numpy.ones((1, 1, n_t), dtype=theano.config.floatX) # will scale out drop-out sample over time to make 3D tensor
        #srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(1000))
        mask = numpy.random.binomial(n=1, p=1-p, size=(n_x, n_s)) #, dtype=theano.config.floatX)
        mask = mask[...,None] * time
        #print("Mask:\n",mask.T)
        return mask.astype(theano.config.floatX)
        #return T.cast(mask, theano.config.floatX)[...,None] * time
    else: # compute expectation
        time = numpy.ones((1, 1, n_t), dtype=theano.config.floatX) # will scale out drop-out sample over time to make 3D tensor
        likeli = numpy.ones((n_x, n_s), dtype=theano.config.floatX) * (1-p)
        return likeli[...,None] * time

def create_zone_out_mask(rng, n_x, n_s, n_t, p, sample=True): # 3D tensor shape:  variable dim x batch-size dim x time-window dim
    if sample and p > 0.0: # compute drop-out masks (i.e., draw samples)
        #srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(1000))
        #mask = srng.binomial(n=1, p=1-p, size=(n_x, n_s, n_t), dtype=theano.config.floatX)
        mask = numpy.random.binomial(n=1, p=1-p, size=(n_x, n_s, n_t))
        return mask.astype(theano.config.floatX)
    else:
        time = numpy.ones((1, 1, n_t), dtype=theano.config.floatX) # will scale out drop-out sample over time to make 3D tensor
        likeli = numpy.ones((n_x, n_s), dtype=theano.config.floatX) * (1-p)
        return (likeli[...,None] * time).astype(theano.config.floatX)
    #return T.cast(mask, theano.config.floatX)
    #output = act * mask
    #output = act*T.cast(mask, theano.config.floatX)
    #return output / (1 - p)
    #return theano.tensor.cast(output / (1 - p), theano.config.floatX)
