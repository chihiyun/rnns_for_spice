# -*- coding: utf-8 -*-
import numpy as np
from numpy import random 
import numpy as np
import math
import datetime
import time
import sys
import models

class DataGenerator: 
    '''
    Read the data from a file.
    start_symbol   : -2 mod |Sigma|
    end_symbol     : -1 mod |Sigma|
    padding_symbol : -1 mod |Sigma|
    '''
    def __init__(self, filename, fmt="pautomac", start_symbol=-2, end_symbol=-1, padding_symbol=-1):
        if fmt is "pautomac":
            tmp = [ l.strip().split()[1:] for l in open(filename).readlines() ][1:]
            if end_symbol is not None:
                tmp = [[start_symbol]+s+[end_symbol] for s in tmp] 
            pos = np.cumsum([0]+[ len(lst) for lst in tmp])
            pos = np.c_[ pos, np.roll(pos,-1)][:-1]
            self.pos = pos
            te = np.arange(len(pos))>(len(pos)*0.99)
            self.pos_tr = pos
            self.pos_te = pos[te]
            self.data = np.asarray([a for lst in tmp for a in lst], dtype=np.int32)
            self.kind = np.max(self.data)+3
            self.data = self.data % self.kind
            self.start_symbol = start_symbol % self.kind
            self.end_symbol = end_symbol % self.kind
            self.padding_symbol = padding_symbol % self.kind


    '''
    Batched letters in batched sentences are yeilded sequentially.
    If the length of sentences are different in the batch,
    We padding the ends of sentences by -1 after shorter sentences are finished.
    arg: batch_ids, an array of ids for sentences.
    return : an array of letters.  
    '''
    def sequence(self, batch_ids, train=True ):
        if train:
            pos = self.pos_tr[batch_ids]
        else:
            pos = self.pos_te[batch_ids]            
        ids = pos[:,0]
        end = pos[:,1] - 1
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
    Called from the inside.
    The sentences are sorted by the length and suffled.
    '''
    def shuffle(self, shuffle_block_size, train):
        if train:
            lengths = self.pos_tr[:,1]-self.pos_tr[:,0]
        else:
            lengths = self.pos_te[:,1]-self.pos_te[:,0]            
        sorted_order =  np.argsort(lengths + np.random.uniform(0, 0.999, len(lengths)) )
        '''When the length of data is not devied by block_size, the set of blocks must have another block.'''
        rem = int( len(lengths) % shuffle_block_size != 0 )
        blocks = np.arange(len(lengths)/shuffle_block_size + rem)
        np.random.shuffle(blocks)
        return blocks, sorted_order
    
    '''
    Yields batched sentences from the starts to the ends simulteniously.  
    return: an array of ids for sentences.
    '''
    def batched_sentences(self, batch_size = 16, train=True ):
        blocks, sorted_order = self.shuffle(batch_size, train)
        for b in blocks:
            yield sorted_order[ b*batch_size : (b+1)*batch_size ]
