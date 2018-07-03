'''
    config file
'''

import tensorflow as tf

# ------------------------------------------------
# File
# ------------------------------------------------
train_file = 'data/dataset1.csv'
valid_file = 'data/dataset2.csv'
test_file = 'data/dataset3.csv'

label_name = 'label'

outputFile_dir = 'DF_submission.csv'
# ------------------------------------------------
# Feature
# ------------------------------------------------
ignore_features = ['user_id', 'label']

numeric_features = ['all_launch_count', 'last_launch', 'all_video_count', 'last_video', 'all_video_day',
                    'all_action_count', 'last_action',
                    'all_action_day', 'register_day']
# ------------------------------------------------
# training
# ------------------------------------------------
show_time = 6  # valid times every epoch
mode = 'valid' # （str : 'valid' | 'test'）  [comment: none]
random_seed = 1995  # （int） [comment: global random seed]
batch_size = 'all'   # (int , str : 'all')  [comment: all' means whole train-set]
epochs = 2000         # (int) [comment: epochs when valid]
test_epochs = 50     # (int) [comment: epochs when train the final model]
deep_layers = [256 ,128, 64, 32] # (list) [ DNN network structure ]
dropout_keep_deep = [0.9, 0.9, 0.9, 0.9, 0.9]  # 1 more than length of deep layers
dropout_keep_fm = [0.8, 0.8]
embedding_size = 32  #
learning_rate = 0.03
batch_norm_decay = 0.995
print_batch = 1 # print per num batchs

train_phase = False
batch_norm = False

l2_reg = 0
# ------------------------------------------------
# model
# ------------------------------------------------
use_fm = True
use_deep = True
output_size = 2  # binary classification

loss_type = 'cross_entropy'  # 'mse' or 'cross_entropy'
deep_layers_activation = 'relu'
optimizer_type = 'adagrad'  # adam adagrad gd momentum yellowfin

# ------------------------------------------------
# momentum mode
# ------------------------------------------------
# [small batch-size + Adagrad, big-batch + momentum , whole-batch + Adagrad]

momentum_mode = True
optimizer_list = ['adam','momentum','adagrad']
batch_size_list = [32,512,'all']
section_epochs = [1,1,1]
print_batch_list = [20,10,1]
recurrent_num = 1








