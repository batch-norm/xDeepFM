'''
    Tools for deepFM
'''

import numpy as np
import pandas as pd
import Config
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from sklearn.metrics import roc_auc_score,f1_score
import tensorflow as tf


# label size example : 2 means [0,1]
def auc_score(preds, labels, label_size):
    preds = [x[label_size - 1] for x in preds]
    labels = [x[label_size - 1] for x in labels]
    roc_score = roc_auc_score(labels, preds)
    return roc_score


def F1_score(preds,labels,label_size,threshold_list):
    max_score = 0
    max_thres = 0
    preds = [x[label_size - 1] for x in preds]
    labels = [x[label_size - 1] for x in labels]
    scores = []
    for thre in threshold_list:
        final_pred = []
        for v in preds:
            if v >= thre:
                final_pred.append(1)
            else:
                final_pred.append(0)
        score = f1_score(labels, final_pred)
        scores.append(score)
        if score > max_score:
            max_score = score
            max_thres = thre
    return max_score

def get_label(labels, label_size):
    final_label = []
    for v in labels:
        temp_label = [0] * label_size
        temp_label[v] = 1
        final_label.append(temp_label)
    return final_label


def get_batch(Xi, Xv, y, batch_size, index):
    if batch_size == 'all':
        return Xi,Xv,get_label(y,2)
    start = index * batch_size
    end = (index + 1) * batch_size
    end = end if end < len(y) else len(y)
    temp_y = [y_ for y_ in y[start:end]]
    temp = get_label(temp_y, 2)
    return Xi[start:end], Xv[start:end], temp

def shuffle_in_unison_scary(a, b, c):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)
    return a, b, c

def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = batch_norm(x, decay=Config.batch_norm_decay, center=True, scale=True, updates_collections=None,
                          is_training=True, reuse=None, trainable=True, scope=scope_bn)
    bn_inference = batch_norm(x, decay=Config.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=False, reuse=True, trainable=True, scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z

def loadData():
    train = pd.read_csv(Config.train_file, index_col=0)
    valid = pd.read_csv(Config.valid_file, index_col=0)
    test = pd.read_csv(Config.test_file, index_col=0)
    return train, valid, test
