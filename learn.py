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
from chainer import serializers
import models
from datagen import DataGenerator
import time
import os

'''
        "args : prob_no, net_arch, VEC_SIZE."
        "net_arch must be chosen from {simple, sp2, bigram}."
'''
def learn( pro_no, net_arch, VEC_SIZE, gpu_no=0):

    print "gpu_no:", gpu_no
    print "problem:", pro_no
    print "net_arch:", net_arch
    print "vec_size:", VEC_SIZE

    try:
        cuda.check_cuda_available()
        cuda.get_device(gpu_no).use()
        import cupy
        print "GPU :"+str(gpu_no)
        xp = cupy
    except:
        xp = np

    time_id = datetime.datetime.today().strftime("%m%d_%H%M")
    dir_name = str.zfill(str(pro_no),2)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    output_file = str.zfill(str(pro_no),2) + '/' +'log_'+ str(VEC_SIZE) + '_' + net_arch + '_' + time_id + '.txt'
    of = open(output_file,'w')

    print >>of, "# net_arch:", net_arch
    print >>of, "# vec_size:", VEC_SIZE

    filename = 'on-line/'+str(pro_no)+'.spice.train'
    data = DataGenerator(filename)

    print output_file
    print "kind   size:", data.kind
    print "data length:", len(data.pos) 
    print "mean length:", np.mean(data.pos[:,1] - data.pos[:,0]) 
    print "max  length:", np.max(data.pos[:,1] - data.pos[:,0]) 

    print data.kind

    if net_arch == "simple":
        learner = models.Simple(vocab_size=data.kind, dim_embed=100, dim1=VEC_SIZE, dim2=VEC_SIZE, dim3=VEC_SIZE/2, class_size=data.kind)
        B_SIZE = 128
    elif net_arch == "sp2":
        learner = models.Sp2(vocab_size=data.kind, dim_embed=100, dim1=VEC_SIZE, dim2=VEC_SIZE, dim3=VEC_SIZE/2, class_size=data.kind)
        B_SIZE = 128
    elif net_arch == "bigram":
        learner = models.Bigram(vocab_size=data.kind, dim_embed=100, dim1=VEC_SIZE, dim2=VEC_SIZE, dim3=VEC_SIZE/2, class_size=data.kind)
        B_SIZE = 96
    else:
        print "net_arch must be chosen from {simple, sp2, bigram}."
        exit -1
 
    whole_epoc = 0

    learner.optimizer = optimizers.MomentumSGD()
    learner.optimizer.setup(learner)

    schedule = [
     [0.1, 5],
     [0.03, 10],
     [0.01, 10],
     [0.003, 10],
     [0.001, 10]
    ]

    while len(schedule) > 0:
        learner.optimizer.lr = schedule[0][0]
        schedule[0][1] -= 1
        if schedule[0][1] <= 0:
            schedule = schedule[1:]
            model_file = str.zfill(str(pro_no),2) + '/' + 'model_' + str.zfill(str(whole_epoc),2) + '_' + str(VEC_SIZE) + '_' + net_arch + '_' + time_id + '.hdf5'
            serializers.save_hdf5(model_file, learner)

        whole_epoc += 1
        for train in [True]:
            s_top5, s_ac, n = 0.0, 0.0, 1
            start = time.time()
            for sid in data.batched_sentences(B_SIZE, train=train):
                #break
                old_xs=None
                for xs, is_valid in data.sequence(sid, train=train):
                    if old_xs is not None:
                        if train:
                            correct, score_top5 = learner.feed_for_learn((xp.asarray(old_xs),xp.asarray(sp2)), xs)
                        else:
                            correct, score_top5 = learner.check((xp.asarray(old_xs),xp.asarray(sp2)), xs)
                        s_top5 += np.sum(score_top5[is_valid])
                        s_ac += np.sum(correct[is_valid])
                        n += np.sum( is_valid )
                        old_xs = np.roll( old_xs, 1, axis=1)
                        old_xs[:,0] = xs 
                        for j,i in enumerate(xs):
                            sp2[j,i] = 1.0
                    else:
                        old_xs = np.zeros((len(xs), 3), dtype=np.int32)
                        old_xs[:,0] = xs
                        sp2 = np.zeros((len(xs), data.kind),dtype=np.float32)
                if train:
                    learner.update()
                else:
                    learner.reset_state()
            end = time.time() - start
            name = {True:'Tr', False:'Te'}
            print '.',
            print >>of, "%02d"%whole_epoc, name[train], 'done:', n,
            print >>of, 'acc:%7.4f'%(s_ac/n), 'top5_scr:%7.4f'%(s_top5/n), 'time:%5.1f'%end,
            if train:
                print >>of, 'lr=', learner.optimizer.lr
            else:
                print >>of, ''
    of.close()
    print ''
    
    model_file = str.zfill(str(pro_no),2) + '/' + 'model_' + str.zfill(str(whole_epoc),2) + '_' + str(VEC_SIZE) + '_' + net_arch + '_' + time_id + '.hdf5'
    serializers.save_hdf5(model_file, learner)


'''
  Starting point of learning loop:
  In all, the following contains:
  Number of trials for statistics : 10
  Number of vector sizes : 2
  Number of different architectures  : 3
  Number of probrem numbers : 16
  If this program is run on single computing node, it may take more than a week.
  If you want to finish in a short time, please change the parameters as appropriate.
'''
if __name__ == '__main__':

    for i in xrange(10):
        for vecsize in [400,600]:
            for arch in ["simple","sp2", "bigram"]:
                for pno in xrange(1,16):
                    start = time.time()
                    learn(pno, arch, vecsize, 0)
                    elapsed_time = time.time() - start
                    print "finished:", i, "th",  (pno, arch, vecsize), ":", elapsed_time, "sec"
