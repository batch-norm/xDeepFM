import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score

def auc_score(preds, labels, label_size):
    preds = [x[label_size - 1] for x in preds]
    labels = [x[label_size - 1] for x in labels]
    roc_score = roc_auc_score(labels, preds)
    return roc_score


def _get_data(data_dir):
    data = []
    with open(data_dir, 'r') as f:
        line = f.readline()
        while line:
            data.append(line)
            line = f.readline()
    return data


def _get_conf():
    with open('data_conf.txt', 'r') as f:
        line = f.readline()
    line = line.split('\t')
    return int(line[0]), int(line[1]), int(line[2]), int(line[3])

def get_label(labels, label_size):
    final_label = []
    for v in labels:
        temp_label = [0] * label_size
        temp_label[v] = 1
        final_label.append(temp_label)
    return final_label
