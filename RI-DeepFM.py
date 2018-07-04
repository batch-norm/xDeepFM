import numpy as np
import pandas as pd
import tensorflow as tf
import time

import Config
from DataReader import FeatureDictionary, DataParser
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from yellowfin import YFOptimizer
from sklearn.metrics import roc_auc_score, f1_score
from tools import auc_score, F1_score, get_label, get_batch, shuffle_in_unison_scary, batch_norm_layer, loadData


# load your own data

class DeepFM(object):
    def __init__(self):
        with tf.Session() as self.sess:
            self.learning_rate = Config.learning_rate
            self.init_data()
            self.init_parameter()
            self.init_model()
            self.init_optmizer()
            self.valid_auc = []
            self.batch_num = []
            self.valid_f1 = []
            self.is_momentum = []
            self.batch_count = 0

            # run
            self.run_model()
            self.save_result()

    def init_data(self):
        # --------------------------------------------------------------------------------------------------------------------------
        #          Read data
        # --------------------------------------------------------------------------------------------------------------------------
        train, valid, test = loadData()
        self.test_id = test.user_id
        self.fd = FeatureDictionary(dfTrain=train, dfTest=test, numeric_cols=Config.numeric_features,
                                    ignore_cols=Config.ignore_features)
        data_parser = DataParser(feat_dict=self.fd)
        self.Xi_train, self.Xv_train, self.y_train = data_parser.parse(df=train, has_label=True)
        self.Xi_valid, self.Xv_valid, self.y_valid = data_parser.parse(df=valid, has_label=True)
        self.Xi_test, self.Xv_test, self.y_test = data_parser.parse(df=test, has_label=True)
        self.y_valid = get_label(self.y_valid, 2)
        self.y_test = get_label(self.y_test, 2)
        self.Xi_train_test = self.Xi_train + self.Xi_valid
        self.Xv_train_test = self.Xv_train + self.Xv_valid
        self.y_train_test = self.y_train + self.y_valid

    def init_parameter(self):
        # --------------------------------------------------------------------------------------------------------------------------
        #          Parameter initialization
        # --------------------------------------------------------------------------------------------------------------------------
        self.field_size = len(self.Xi_train[0])
        input_size = self.field_size * Config.embedding_size
        glorot = np.sqrt(2.0 / (input_size + Config.deep_layers[0]))

        tf.set_random_seed(seed=Config.random_seed)
        self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name="feat_index")
        self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name="feat_value")
        self.label = tf.placeholder(tf.float32, shape=[None, 2], name="label")
        self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
        self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
        self.train_phase = tf.placeholder(tf.bool, name="train_phase")
        self.weights = dict()

        self.weights["feature_embeddings"] = tf.Variable(
            # tf.random_normal([self.fd.feat_dim, Config.embedding_size], 0.0, 0.01), name="feature_embeddings"
            np.random.normal(loc=0, scale=glorot, size=(self.fd.feat_dim, Config.embedding_size)), dtype=np.float32
        )
        self.weights["feature_bias"] = tf.Variable(tf.random_uniform([self.fd.feat_dim, 1], 0.0, 1.0),
                                                   name="feature_bias")
        # deep layers
        num_layer = len(Config.deep_layers)

        self.weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, Config.deep_layers[0])), dtype=np.float32)
        self.weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, Config.deep_layers[0])),
                                             dtype=np.float32)
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (Config.deep_layers[i - 1] + Config.deep_layers[i]))
            self.weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(Config.deep_layers[i - 1], Config.deep_layers[i])),
                dtype=np.float32)
            self.weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, Config.deep_layers[i])),
                dtype=np.float32)
        if Config.use_fm and Config.use_deep:
            input_size = self.field_size + Config.embedding_size + Config.deep_layers[-1]
        elif Config.use_fm:
            input_size = self.field_size + Config.embedding_size
        elif Config.use_deep:
            input_size = Config.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        self.weights["concat_projection"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, Config.output_size)), dtype=np.float32)
        self.weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

    def init_model(self):
        # --------------------------------------------------------------------------------------------------------------------------
        #          Deep + FM Model
        # --------------------------------------------------------------------------------------------------------------------------
        embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"], self.feat_index)
        input_feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])
        embeddings = tf.multiply(embeddings, input_feat_value)
        y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index)
        y_first_order = tf.reduce_sum(tf.multiply(y_first_order, input_feat_value), 2)
        if Config.batch_norm:
            y_first_order = batch_norm_layer(y_first_order, train_phase=self.train_phase,scope_bn="bn_first")
        y_first_order = tf.nn.dropout(y_first_order, Config.dropout_keep_fm[0])

        summed_features_emb = tf.reduce_sum(embeddings, 1)
        summed_features_emb_square = tf.square(summed_features_emb)
        squared_features_emb = tf.square(embeddings)
        squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)
        y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)
        if Config.batch_norm:
            y_second_order = batch_norm_layer(y_second_order, train_phase=self.train_phase,scope_bn='bn_second')
        y_second_order = tf.nn.dropout(y_second_order, Config.dropout_keep_fm[1])

        y_deep = tf.reshape(embeddings, shape=[-1, self.field_size * Config.embedding_size])
        y_deep = tf.nn.dropout(y_deep, Config.dropout_keep_deep[0])
        for i in range(0, len(Config.deep_layers)):
            y_deep = tf.add(tf.matmul(y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])
            if Config.batch_norm:
                y_deep = batch_norm_layer(y_deep, train_phase=self.train_phase, scope_bn="bn_%d" % i)
            if Config.deep_layers_activation == 'sigmoid':
                y_deep = tf.nn.sigmoid(y_deep)
            else:
                y_deep = tf.nn.relu(y_deep)
            y_deep = tf.nn.dropout(y_deep, self.dropout_keep_deep[1 + i])  # dropout at each Deep layer
        concat_input = None
        # output
        if Config.use_fm and Config.use_deep:
            concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
        elif Config.use_fm:
            concat_input = tf.concat([y_first_order, y_second_order], axis=1)
        elif Config.use_deep:
            concat_input = y_deep
        self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])
        self.softmax_out = tf.nn.softmax(self.out)
        # loss
        if Config.loss_type == "mse":
            self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
        else:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.out),
                                       name='loss')
        # l2
        if Config.l2_reg > 0:
            self.loss += tf.contrib.layers.l2_regularizer(Config.l2_reg)(self.weights["concat_projection"])
            if Config.use_deep:
                for i in range(len(Config.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(
                        Config.l2_reg)(self.weights["layer_%d" % i])
        # optimizer

    def init_optmizer(self):
        self.Adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1e-8).minimize(self.loss)
        self.Adagrad = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.GD = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.Momentum = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
            self.loss)
        self.YF = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(self.loss)

    def get_optimizer(self, name):
        if name == 'adam':
            return self.Adam
        elif name == 'adagrad':
            return self.Adagrad
        elif name == 'gd':
            return self.GD
        elif name == 'momentum':
            return self.Momentum
        else:
            return self.YF

    # --------------------------------------------------------------------------------------------------------------------------
    #          Run
    # --------------------------------------------------------------------------------------------------------------------------
    def run_model(self):
        start = time.time()
        mode = Config.mode
        if mode is not 'valid' and mode is not 'test':
            print('reset the config -> mode!')
        self.sess.run(tf.global_variables_initializer())
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("#params: %d" % total_parameters)

        # Recurrent
        if Config.momentum_mode == True:
            for i in range(Config.recurrent_num):
                for j in range(len(Config.section_epochs)):
                    print('-------------------------------------------------------------')
                    print('recurrent_num:%d' % (i + 1),
                          'epochs:%d' % Config.section_epochs[j],
                          'batch_size:%d' % Config.batch_size_list[j],
                          'optimizer:%s' % Config.optimizer_list[j],
                          'learning_rate:%.4f' % Config.learning_rate_list[j])
                    print('-------------------------------------------------------------')
                    self.learning_rate = Config.learning_rate_list[j]
                    if mode == 'valid':
                        self.run_valid(Config.section_epochs[j], self.get_optimizer(Config.optimizer_list[j]),
                                       Config.batch_size_list[j], Config.print_batch_list[j],Config.optimizer_list[j])
                    # elif mode == 'test':

        # Normal
        else:
            epochs_list = [Config.epochs]
            for epochs in epochs_list:
                if mode == 'valid':
                    self.run_valid(epochs, self.get_optimizer(Config.optimizer_type), Config.batch_size,
                                   Config.print_batch,'hello')
                # elif mode == 'test':
        end = time.time()
        print('total time:',end-start)

    # --------------------------------------------------------------------------------------------------------------------------
    #          valid
    # --------------------------------------------------------------------------------------------------------------------------

    def run_valid(self, epochs, optimizer, batch_size, print_batch,optimizer_name):
        total_batch = int(len(self.y_train) / batch_size)
        all_batchs = epochs * total_batch
        count = 0
        for i in range(epochs):
            # shuffle every epoch
            self.Xi_train, self.Xv_train, self.y_train = shuffle_in_unison_scary(self.Xi_train, self.Xv_train,
                                                                                 self.y_train)
            for j in range(total_batch):
                # train
                Xi_batch, Xv_batch, y_batch = get_batch(self.Xi_train, self.Xv_train, self.y_train, batch_size,
                                                        j)
                feed_dict = {self.feat_index: Xi_batch,
                             self.feat_value: Xv_batch,
                             self.label: y_batch,
                             self.dropout_keep_fm: Config.dropout_keep_fm,
                             self.dropout_keep_deep: Config.dropout_keep_deep,
                             self.train_phase: True}
                train_softmax, train_loss_value, train_opt, train_output = self.sess.run(
                    (self.softmax_out, self.loss, optimizer, self.out), feed_dict=feed_dict)

                # valid
                count += 1
                self.batch_count += 1
                if count % print_batch == 0:
                    valid_feed = {
                        self.feat_index: self.Xi_valid,
                        self.feat_value: self.Xv_valid,
                        self.label: self.y_valid,
                        self.dropout_keep_fm: [1.0] * len(Config.dropout_keep_fm),
                        self.dropout_keep_deep: [1.0] * len(Config.dropout_keep_deep),
                        self.train_phase: False
                    }
                    valid_softmax, valid_loss_value, valid_output = self.sess.run(
                        (self.softmax_out, self.loss, self.out), feed_dict=valid_feed)
                    print(
                          'epochs:%d' % (i + 1),
                          'total_batch:%d' % all_batchs,
                          'batch:%d' % count,
                          'logloss:%f' % train_loss_value,
                          'valid_loss:%f' % valid_loss_value,
                          'valid_auc:%.4f' % auc_score(valid_softmax, self.y_valid, 2),
                          'valid_f1:%f' % F1_score(valid_softmax, self.y_valid, 2, [0.39, 0.4, 0.41])
                          )
                    self.valid_auc.append(auc_score(valid_softmax, self.y_valid, 2))
                    self.batch_num.append(self.batch_count)
                    self.valid_f1.append(F1_score(valid_softmax, self.y_valid, 2, [0.39, 0.4, 0.41]))
                    if optimizer_name == 'momentum':
                       self.is_momentum.append(1)
                    else:
                        self.is_momentum.append(0)

    # --------------------------------------------------------------------------------------------------------------------------
    #          test
    # --------------------------------------------------------------------------------------------------------------------------
    # def run_test(self, epochs, optimizer,batch_size,print_batch):
    #     # train
    #     total_batch = int(len(self.y_train) / batch_size)
    #     all_batchs = epochs * total_batch
    #     for i in range(Config.test_epochs):
    #         # shuffle every epoch
    #         Xi_train, Xv_train, y_train = shuffle_in_unison_scary(self.Xi_train_test, self.Xv_train_test, self.y_train_test)
    #         for j in range(total_batch):
    #             Xi_batch, Xv_batch, y_batch = get_batch(Xi_train, Xv_train, y_train, Config.batch_size, j)
    #             feed_dict = {self.feat_index: Xi_batch,
    #                          self.feat_value: Xv_batch,
    #                          self.label: y_batch,
    #                          self.dropout_keep_fm: Config.dropout_keep_fm,
    #                          self.dropout_keep_deep: Config.dropout_keep_deep,
    #                          self.train_phase: True}
    #             train_softmax, train_loss_value, train_opt, train_output = self.sess.run(
    #                 (self.softmax_out, self.loss, optimizer, self.out),
    #                 feed_dict=feed_dict)
    #             if j % 10 == 0:
    #                 print('epochs:%d' % i, 'logloss:%f' % train_loss_value)
    #     # predict
    #     feed_dict = {self.feat_index: self.Xi_test,
    #                  self.feat_value: self.Xv_test,
    #                  self.label: self.y_test,
    #                  self.dropout_keep_fm: [1.0] * len(Config.dropout_keep_fm),
    #                  self.dropout_keep_deep: [1.0] * len(Config.dropout_keep_deep),
    #                  self.train_phase: False}
    #     test_out = self.sess.run(self.softmax_out, feed_dict=feed_dict)
    #     test_out = [x[1] for x in test_out]
    #     submission = pd.DataFrame()
    #     print(len(self.test_id))
    #     print(len(test_out))
    #     submission['user_id'] = self.test_id
    #     submission['prob'] = test_out
    #     submission = submission.sort_values(by='prob', ascending=False)
    #     submission.to_csv(Config.outputFile_dir, index=False)

    def save_result(self):
        result = pd.DataFrame()
        result['batch_num'] = self.batch_num
        result['valid_auc'] = self.valid_auc
        result['valid_f1'] = self.valid_f1
        result['is_momen'] = self.is_momentum
        result.to_csv('train_process.csv',index=False)


if __name__ == '__main__':
    dm = DeepFM()
