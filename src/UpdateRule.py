"""
The MIT License (MIT)
Copyright (c) 2015 Alec Radford
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# I borrowed this bit of code from my colleague/collaborator Iulian =]
# Why re-invent this adaptive learning rate "wheel"?

import theano
from theano.ifelse import ifelse
import theano.tensor as T
from collections import OrderedDict

def sharedX(value, name=None, borrow=False, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def Adam(grads, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, norm_max=-1.0):
    updates = []
    i = sharedX(0.)
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in grads.items():
        m = sharedX(p.get_value() * 0.)
        v = sharedX(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        if norm_max > 0.0: # apply parameter reprojection if desired
            p_norm = p_t.norm(2)
            p_t = ifelse(T.gt(p_norm, norm_max), p_t*(norm_max / p_norm), p_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def RMSprop(grads, lr=0.001, decay=0.95, eta=0.9, epsilon=1e-6, norm_max=-1.0): 
    updates = OrderedDict()
    for param in grads.keys():
        # mean_squared_grad := E[g^2]_{t-1}
        mean_square_grad = sharedX(param.get_value() * 0.)
        mean_grad = sharedX(param.get_value() * 0.)
        delta_grad = sharedX(param.get_value() * 0.)
        if param.name is None:
            raise ValueError("Model parameters must be named.")
        mean_square_grad.name = 'mean_square_grad_' + param.name
        # Accumulate gradient
        new_mean_grad = (decay * mean_grad + (1 - decay) * grads[param])
        new_mean_squared_grad = (decay * mean_square_grad + (1 - decay) * T.sqr(grads[param]))
        # Compute update 
        scaled_grad = grads[param] / T.sqrt(new_mean_squared_grad - new_mean_grad ** 2 + epsilon)
        new_delta_grad = eta * delta_grad - lr * scaled_grad 
        # Apply update
        updates[delta_grad] = new_delta_grad
        updates[mean_grad] = new_mean_grad
        updates[mean_square_grad] = new_mean_squared_grad
        p_t = param + new_delta_grad
        if norm_max > 0.0: # apply parameter reprojection if desired
            p_norm = param.norm(2)
            p_t = ifelse(T.gt(p_norm, norm_max), p_t*(norm_max / p_norm), p_t)
        updates[param] = p_t
    return updates
