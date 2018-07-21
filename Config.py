"""
    author : yang yiqing 2018年07月13日15:57:50
"""

# file
train_file = 'ex_data/dataset1.csv'
valid_file = 'ex_data/dataset2.csv'
test_file = 'ex_data/dataset3.csv'

train_save_file = 'ex_data/dataset1.txt'
valid_save_file = 'ex_data/dataset2.txt'
test_save_file = 'ex_data/dataset3.txt'

label_name = 'label'

# features
numeric_features = ['all_launch_count', 'last_launch', 'all_video_count', 'last_video', 'all_video_day',
                    'all_action_count', 'last_action',
                    'all_action_day', 'register_day']
single_features = ['register_type', 'device_type']
multi_features = []

num_embedding = True
single_feature_frequency = 10
multi_feature_frequency = 0

# model

FM_layer = True
DNN_layer = True
CIN_layer = False

use_numerical_embedding = False


embedding_size = 16

dnn_net_size = [128,64,32]
cross_layer_size = [10,10,10]
cross_direct = False
cross_output_size = 1

# train
batch_size = 4096
epochs = 4000
learning_rate = 0.01



