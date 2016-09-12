# -*- coding: utf-8 -*-
import numpy as np
from numpy import random 
import numpy as np
import math
import datetime
import time
import sys
import lstms
import datagen
import os
import glob
from chainer import serializers
from chainer import cuda
import itertools as ite

class TestDataGenerator: 
    '''
    pautomac 形式の時系列のデータファイルを読み込みます。
    start_symbol   : -2 mod 記号の種類数
    end_symbol     : -1 mod 記号の種類数
    padding_symbol : -1 mod 記号の種類数
    '''
    def __init__(self, kind, prefix_file, target_file, start_symbol=-2, end_symbol=-1, padding_symbol=-1):
        '''prefix ファイルの読み込み'''
        tmp = [ l.strip().split()[1:] for l in open(prefix_file).readlines() ][1:]
        self.datasize = len(tmp)
        if end_symbol is not None:
            tmp = [[start_symbol]+s+[end_symbol] for s in tmp] 
        pos = np.cumsum([0]+[ len(lst) for lst in tmp])
        pos = np.c_[ pos, np.roll(pos,-1)][:-1]
        self.pos = pos
        self.data = np.asarray([a for lst in tmp for a in lst], dtype=np.int32)
        self.kind = kind
        self.data = self.data % self.kind
        self.start_symbol = start_symbol % self.kind
        self.end_symbol = end_symbol % self.kind
        self.padding_symbol = padding_symbol % self.kind
        
        '''ターゲットファイルの読み込み'''
        self.denominators = np.zeros(self.datasize, dtype=float)
        self.wd_to_ps = np.zeros( (self.datasize,self.kind), dtype=float)
        for i, l in enumerate( open(target_file).readlines()[:self.datasize] ) :
            ary  = l.strip().split()
            self.denominators[i] = float(ary[0])
            mat = np.asarray(ary[1:]).reshape(-1,2)
            wds = mat[:,0].astype(int)
            wds[wds >= self.kind] = self.kind-1
            wds = wds % self.kind
            ps = mat[:,1].astype(float)
            self.wd_to_ps[i, wds ] = ps
    '''
    バッチ化された時系列データを、順に yield で生成します。
    テストなので、時系列の長さが異なることは想定していません。
    引数: batch_ids  バッチ化するための、時系列のIDの配列
    '''
    def sequence(self, batch_ids):
        ids = self.pos[batch_ids,0]
        end = self.pos[batch_ids,1] - 1
        xs = self.data[ids]
        yield xs.copy(), np.ones(len(xs), dtype=np.bool)
        dif = ids < end
        while np.any(dif) :
            ids = ids + dif
            xs[~dif] = self.padding_symbol
            xs[dif] = self.data[ids][dif]
            yield xs.copy(), dif
            dif = ids < end
            
    '''
    時系列データのセンテンスの配列(バッチ)を、順に yield で生成します。
    長さを揃えます。
    '''
    def batched_sentences(self, max_batch_size ):
        '''prefix sentences を block に分割'''
        lengths = self.pos[:,1]-self.pos[:,0]
        sorted_order =  np.argsort(lengths + np.arange(len(lengths))*(0.1/(len(lengths))) )
        #print lengths[sorted_order]
        '''ひとつ前より真に大きいならTrue'''
        flag = np.roll(lengths[sorted_order], 1) < lengths[sorted_order]
        #print flag
        '''Trueの累積和 = block_IDs'''
        block_ids = flag.cumsum()
        splited_orders = [sorted_order[block_ids == i] for i in range(0, np.max(block_ids)+1)]
        '''さらに max_batch_size で割る'''
        blocks = [ ary[i:i+max_batch_size] for ary in splited_orders for i in range(0,len(ary),max_batch_size) ]
        '''block を yield '''
        for sen_ids in blocks:
            yield np.asarray(sen_ids, dtype=int)
    


#print data.kind
try:
    cuda.check_cuda_available()
    cuda.get_device(1).use()
    import cupy
    xp = cupy
    print "use cupy as xp"
except:
    print "use numpy as xp"
    xp = np

def target_score(self, top5s, sids):
    #print 'sids', sids+1
    #print 'top5s', top5s
    #for sid in sids: print sid +1, self.wd_to_ps[sid] 
    mat = np.asarray([ self.wd_to_ps[sid, top5] for top5, sid in zip( top5s, sids) ]) 
    norms = self.denominators[sids]
    #print mat
    mat = mat /np.log2([2,3,4,5,6])
    #print mat
    srcs = np.sum( mat , axis=1 )
    #print 'srcs', srcs
    #print 'norms', norms
    scores_top5 =  srcs/ norms
    accs = mat[:,0]
    return accs, scores_top5

def print_test_score(learner, test_data):

    start = time.time()
    corrects, scores=[], []
    for sid in test_data.batched_sentences(64):
        learner.reset_state()
        old_xs=None
        checked = False
        for xs, is_valid in test_data.sequence(sid):
            xs %= data.kind
            if old_xs is not None:
                if np.all(xs[0] == -1%data.kind):
                    assert np.all(xs[0] == -1%data.kind), "arignment of sentence lengthes."
                    checked = True
                    correct, score_top5 = learner.check((xp.asarray(old_xs),xp.asarray(sp2)), xs)
                else:
                    learner.feed((xp.asarray(old_xs),xp.asarray(sp2)))
                old_xs = np.roll( old_xs, 1, axis=1)
                old_xs[:,0] = xs 
                for j,i in enumerate(xs):
                    sp2[j,i] = 1.0
            else:
                old_xs = np.zeros((len(xs), 3), dtype=np.int32)
                old_xs[:,0] = xs
                sp2 = np.zeros((len(xs), data.kind),dtype=np.float32)

        assert checked, "no words are predected."
        #print ".",

        accs, scs = target_score(test_data, learner.top5s, sid)
        scores.extend( scs.tolist() )
        corrects.extend( accs.tolist() )

    end = time.time() - start
    print  'acc:%7.4f'%np.mean(corrects), 'top5_scr:%7.4f'%np.mean(scores), 'time:%5.1f'%end


def make_learner(kind, vsize, arch):
    if arch == "simple":
        learner = lstms.Simple(vocab_size=kind, dim_embed=100, dim1=vsize, dim2=vsize, dim3=vsize/2, class_size=kind)
    elif arch == "sp2":
        learner = lstms.Sp2(vocab_size=kind, dim_embed=100, dim1=vsize, dim2=vsize, dim3=vsize/2, class_size=kind)
    elif arch == "bigram":
        learner = lstms.Bigram(vocab_size=kind, dim_embed=100, dim1=vsize, dim2=vsize, dim3=vsize/2, class_size=kind)
    else:
        print "net_arch error"
    return learner

pro_no = range(1,5)+range(6,16)#[13,14,15]#range(4,16)
vec_size = [400,600]
net_arch = ["sp2","simple","bigram"]

for pno in pro_no:
    data =  datagen.DataGenerator('../on-line/'+str(pno)+'.spice.train')
    test_data = TestDataGenerator(data.kind,
                                  '../on-line/prefixes/'+str(pno)+'.spice.prefix.public',
                                  '../on-line/targets/' +str(pno)+'.spice.target.public')
    for vsize, arch in ite.product(vec_size, net_arch):

        model_names_old  = glob.glob( '%s/model_45_%d_%s_????_????.hdf5'%(str(pno).zfill(2),vsize, arch))
        model_names_new = glob.glob( '../%s/model_45_%d_%s_????_????.hdf5'%(str(pno).zfill(2),vsize, arch))
        model_names =  model_names_old + model_names_new 
        if len(model_names) == 0:
            print "glob error:model_final_p_"+ str(pno).zfill(2) +"_*.hdf5"
            print model_names
        else:
            print len(model_names), '×', model_names[0][:-15] 
            for name in model_names[:10]:
                if name in model_names_old:
                    log_name = str(pno).zfill(2)+'/log'+name[11:-5]+'.txt'
                else:
                    log_name = '../'+str(pno).zfill(2)+'/log'+name[11+3:-5]+'.txt'                    
                train_score_top5 = open(log_name,'r').readlines()[-1].strip().split()[7]

                print name[-5-9:-5], "train_score_top5:", train_score_top5,
                learner = make_learner(data.kind, vsize, arch)
                serializers.load_hdf5(name, learner)
                print_test_score(learner, test_data)    

