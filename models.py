# -*- coding: utf-8 -*-
import numpy as np
from numpy import random 
from chainer import cuda, Chain, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
import chainer.links as L
import numpy as np
import math
import datetime
import time
from chainer.functions.array import *
from chainer.functions.math import matmul
import sys

try:
    cuda.check_cuda_available()
    import cupy
    xp = cupy
except:
    xp = np

'''
The abstract class.
'''
class LSTMBase(Chain):
    def update(self):
        self.optimizer.zero_grads()
        self.loss_var.backward() # calc grads
        self.optimizer.clip_grads(5.0) # truncate
        self.optimizer.update()
        self.reset_state()
        
    def score(self, ys, ts):
        ret = []
        top5s = np.argsort(ys, axis=1)[:,::-1][:,:5]
        self.top5s = top5s
        for t,top5 in zip(ts,top5s):
            if t in top5:
                j = np.nonzero(top5 == t)[0] +1
                ret.append( 1.0/np.log2(j+1) )
            else:
                ret.append( 0.0 )
        return np.asarray(ret,dtype=np.float32)

    def feed_for_learn(self, xs, ts):
        xss, tss = xs, Variable(xp.asarray(ts))
        yss = self(xss, train=True)
        loss_tmp = F.softmax_cross_entropy(yss, tss)
        out = loss_tmp.creator.inputs[0].data
        if xp is not np : out = out.get() 
        corrects = ( np.argmax(out,axis=1) == ts )
        score_top5 = self.score(out, ts)
        self.loss_var += loss_tmp
        return corrects, score_top5

    def only_feed_for_learn(self, xs):
        self(xs, train=True)
        return

    def feed(self, xs ):
        # volatile = True -> BP chains are teard -> a test don't use BPs -> rapid calculation 
        xss = xs
        yss = self(xss, train=False)
        return yss, F.softmax(yss).data

    def check(self, xs, ts):
        yss, out = self.feed( xs ) 
        if xp is not np : out =out.get() 
        ts = np.asarray ( ts )
        ac = F.accuracy(yss,  Variable(xp.asarray(ts),volatile = True)).data.tolist()
        corrects = ( np.argmax(out,axis=1) == ts )
        score_top5 = self.score(out, ts)
        return corrects, score_top5

    def only_feed(self, xs):
        # volatile = True -> BP chains are teard -> a test don't use BPs -> rapid calculation 
        xss = Variable(xp.asarray(xs),volatile = True)
        self(xss, train=False)
        return


'''
Simple LSTM model.
'''
class Simple(LSTMBase):
    def __init__(self, vocab_size, dim_embed=33*3, dim1=400, dim2=400, dim3=200, class_size=None):
        if class_size is None:
            class_size = vocab_size
        super(Simple, self).__init__(
            embed2 = L.EmbedID(vocab_size, dim_embed),
            lay2    = L.LSTM(dim_embed, dim1, forget_bias_init=0),
            lay_int = L.LSTM(dim1, dim2, forget_bias_init=0),
            lin1    = L.Linear(dim2,    dim3),
            lin2    = L.Linear(dim3,    class_size),
        )
        self.vocab_size = vocab_size
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print 'run on the GPU.'
        except:
            print 'run on the CPU.'
        self.dim_embed = dim_embed
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))
        self.reset_state()

    def __call__(self, xs, train):
        x_3gram = xs[0]
        sp2 = xs[1]
        
        x_uni = x_3gram[:,0]
        y = Variable(x_uni, volatile = not train)
        y = self.embed2(y)     
        y2 = self.lay2(y)
        y2 = self.lay_int(y2)                
        y = y2
        y = self.lin1(F.dropout(y, train=train))
        y = F.relu(y)
        y = self.lin2(F.dropout(y, train=train)) 

        return y

    def reset_state(self):
        if self.loss_var is not None:
            self.loss_var.unchain_backward()  # for safty
        self.loss_var = Variable(xp.zeros((), dtype=xp.float32)) # reset loss_var
        self.lay2.reset_state()
        self.lay_int.reset_state()
        return

'''
The model using Sp2.
'''
class Sp2(LSTMBase):
    def __init__(self, vocab_size, dim_embed=33*3, dim1=400, dim2=400, dim3=200, class_size=None):
        if class_size is None:
            class_size = vocab_size
        super(Sp2, self).__init__(
            embed2 = L.EmbedID(vocab_size, dim_embed),
            embed3 = L.Linear(vocab_size, dim_embed),
            lin3   = L.Linear(dim_embed, dim1),
            lay2    = L.LSTM(dim_embed, dim1, forget_bias_init=0),
            lay_int = L.LSTM(dim1, dim2, forget_bias_init=0),
            lin1    = L.Linear(dim1+dim2,    dim3),
            lin2    = L.Linear(dim3,    class_size),
        )
        self.vocab_size = vocab_size
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print 'run on the GPU.'
        except:
            print 'run on the CPU.'
        self.dim_embed = dim_embed
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))
        self.reset_state()

    def __call__(self, xs, train):
        x_3gram = xs[0]
        sp2 = xs[1]
        
        x_uni = x_3gram[:,0]
        y = Variable(x_uni, volatile = not train)
        y = self.embed2(y)     
        y2 = self.lay2(y)
        y2 = self.lay_int(y2)        

        y = Variable(sp2, volatile = not train)
        y = self.embed3(y)
        y = self.lin3(y)
        y3 = F.relu(y)
        
        y = concat.concat((y2,y3) )
        y = self.lin1(F.dropout(y, train=train))
        y = F.relu(y)
        y = self.lin2(F.dropout(y, train=train)) 
        
        return y

    def reset_state(self):
        if self.loss_var is not None:
            self.loss_var.unchain_backward()  # 念のため
        self.loss_var = Variable(xp.zeros((), dtype=xp.float32)) # reset loss_var
        self.lay2.reset_state()
        self.lay_int.reset_state()
        return

'''
The model using Bigrams.
'''
class Bigram(LSTMBase):
    def __init__(self, vocab_size, dim_embed=33*3, dim1=400, dim2=400, dim3=200, class_size=None):
        if class_size is None:
            class_size = vocab_size
        super(Bigram, self).__init__(
            embed_uni = L.EmbedID(vocab_size, dim_embed),
            embed_bi  = L.EmbedID(vocab_size*vocab_size, dim_embed),
            lay_uni   = L.LSTM(dim_embed, dim1, forget_bias_init=0),
            lay_bi    = L.StatelessLSTM(dim_embed, dim1, forget_bias_init=0),
            lay_int = L.LSTM(dim1*3, dim2, forget_bias_init=0),
            lin1    = L.Linear(dim2,    dim3),
            lin2    = L.Linear(dim3,    class_size),
        )
        self.vocab_size = vocab_size
        try:
            cuda.check_cuda_available()
            self.to_gpu()
            print 'run on the GPU.'
        except:
            print 'run on the CPU.'
        self.dim_embed = dim_embed
        self.optimizer = optimizers.MomentumSGD()
        self.optimizer.setup(self)
        self.loss_var = Variable(xp.zeros((), dtype=np.float32))
        self.reset_state()

    def __call__(self, xs, train):

        x_3gram = xs[0]
        sp2 = xs[1]
        
        x_uni = x_3gram[:,0]
        y = Variable(x_uni, volatile = not train)
        y = self.embed_uni(y)     
        y_uni = self.lay_uni(y)

        ## bigram です。
        x_bi = x_3gram[:,0]*self.vocab_size + x_3gram[:,1]
        y = Variable(x_bi, volatile = not train)
        y = self.embed_bi(y)
        if self.is_odd:
            self.c_odd, self.h_odd = self.lay_bi(self.c_odd, self.h_odd, y)
            if self.h_evn is None:
                self.h_evn = Variable(xp.zeros_like(self.h_odd.data), volatile = not train)
            y = concat.concat((y_uni, self.h_odd, self.h_evn) )
        else:
            self.c_evn, self.h_evn = self.lay_bi(self.c_evn, self.h_evn, y)
            y = concat.concat((y_uni, self.h_evn, self.h_odd) )
        self.is_odd = not self.is_odd 
        
        y = self.lay_int(y)
        y = self.lin1(F.dropout(y, train=train))
        y = F.relu(y)
        y = self.lin2(F.dropout(y, train=train)) 
        
        return y

    def reset_state(self):
        if self.loss_var is not None:
            self.loss_var.unchain_backward()  # 念のため
        self.loss_var = Variable(xp.zeros((), dtype=xp.float32)) # reset loss_var
        self.lay_uni.reset_state()
        self.is_odd= True
        self.c_odd = None
        self.c_evn = None
        self.h_odd = None
        self.h_evn = None
        self.lay_int.reset_state()
        return
