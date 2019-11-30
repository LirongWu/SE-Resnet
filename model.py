#-*- coding: UTF-8 -*-
import os
import numpy as np
import tensorflow as tf

from cifar10 import *
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib.layers import batch_norm, flatten

class Model:

    def __init__(self, params):

        self.dataset_path = params.dataset_path

        self.learning_rate = params.lr
        self.momentum = params.momentum

        self.batch_size = params.batch_size
        self.total_epochs = params.n_epoch
        self.ratio = params.ratio
        self.mode = params.mode
        self.training = True
        self.weight_decay = 0.0005
        
        #os.environ["CUDA_VISIBLE_DEVICES"] = params.n_gpu
        


    def Relu(self, x):
        return tf.nn.relu(x)

    def Sigmoid(self, x) :
        return tf.nn.sigmoid(x)

    def Global_Average_Pooling(self, x):
        return tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    def Batch_Normalization(self, x, training, scope):
        with arg_scope([batch_norm],
                    scope=scope,
                    updates_collections=None,
                    decay=0.9,
                    center=True,
                    scale=True,
                    zero_debias_moving_mean=True) :
            return tf.cond(training,
                        lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                        lambda : batch_norm(inputs=x, is_training=training, reuse=True))

    def conv_layer(self, x, filters, kernel, stride, padding='SAME', layer_name="Conv"):
        with tf.name_scope(layer_name):
            x = tf.layers.conv2d(x, use_bias=False, filters=filters, kernel_size=kernel, strides=stride, padding=padding)
            return x

    def Fully_connected(self, x, units, layer_name='Fully_connected') :
        with tf.name_scope(layer_name) :
            return tf.layers.dense(inputs=x, units=units)

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = self.conv_layer(x=x, filters=64, kernel=3, stride=1, layer_name=scope+'_conv1')
            x = self.Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = self.Relu(x)

            return x

    def transform_layer(self, x, stride, out_dim, scope):
        with tf.name_scope(scope) :
            x = self.conv_layer(x, filters=out_dim/4, kernel=1, stride=stride, layer_name=scope+'_conv1')
            x = self.Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = self.Relu(x)

            x = self.conv_layer(x, filters=out_dim/4, kernel=3, stride=1, layer_name=scope+'_conv2')
            x = self.Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = self.Relu(x)

            x = self.conv_layer(x, filters=out_dim, kernel=1, stride=1, layer_name=scope+'_conv3')
            x = self.Batch_Normalization(x, training=self.training, scope=scope+'_batch3')

            return x

    def transistion_layer(self, x, stride, out_dim, scope):
        with tf.name_scope(scope) :
            x = self.conv_layer(x, filters=out_dim, kernel=1, stride=stride, layer_name=scope+'_conv1')
            x = self.Batch_Normalization(x, training=self.training, scope=scope+'_batch1')

            return x

    def se_block(self, x, out_dim, scope):
        with tf.name_scope(scope) :
            squeeze = self.Global_Average_Pooling(x)

            excitation = self.Fully_connected(squeeze, units=out_dim / self.ratio, layer_name=scope+'_fully_connected1')
            excitation = self.Relu(excitation)
            excitation = self.Fully_connected(excitation, units=out_dim, layer_name=scope+'_fully_connected2')
            excitation = self.Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = x * excitation

            return scale

    def nl_block(self, x, out_dim, scope):
        with tf.name_scope(scope) :
            n, h, w, c = x.get_shape().as_list()

            wv = self.conv_layer(x, filters=out_dim, kernel=1, stride=1, layer_name=scope+'_conv1')
            wk = self.conv_layer(x, filters=out_dim, kernel=1, stride=1, layer_name=scope+'_conv2')
            wq = self.conv_layer(x, filters=out_dim, kernel=1, stride=1, layer_name=scope+'_conv3')

            wv = tf.transpose(wv, [0,3,1,2])
            wk = tf.transpose(wk, [0,3,1,2])
            wq = tf.transpose(wq, [0,3,1,2])

            wv = tf.reshape(wv, [n, out_dim, -1])
            wq = tf.reshape(wq, [n, out_dim, -1])
            wk = tf.reshape(wk, [n, out_dim, -1])
            wk = tf.transpose(wk, [0,2,1])

            f = tf.matmul(wk, wq)
            f_softmax = tf.nn.softmax(f, 1)

            y = tf.matmul(wv, f_softmax)
            y = tf.reshape(y, [n, out_dim, h, w])
            y = tf.transpose(y, [0,2,3,1])

            wz = self.conv_layer(y, filters=out_dim, kernel=1, stride=1, layer_name=scope+'_conv4')
            z = x + wz

            return z

    def snl_block(self, x, out_dim, scope):
        with tf.name_scope(scope) :
            n, h, w, c = x.get_shape().as_list()

            wk = self.conv_layer(x, filters=1, kernel=1, stride=1, layer_name=scope+'_conv1')
            wk = tf.reshape(wk, [n, h*w, 1])

            wk_softmax = tf.nn.softmax(wk, 1)

            x_reshape = tf.transpose(x, [0,3,1,2])
            x_reshape = tf.reshape(x_reshape, [n, c, -1])
            f = tf.matmul(x_reshape, wk)
            y = tf.reshape(f, [n, 1, 1, out_dim])

            wz = self.conv_layer(y, filters=out_dim, kernel=1, stride=1, layer_name=scope+'_conv2')
            z = x + wz

            return z

    def gc_block(self, x, out_dim, scope):
        with tf.name_scope(scope) :
            n, h, w, c = x.get_shape().as_list()

            wk = self.conv_layer(x, filters=1, kernel=1, stride=1, layer_name=scope+'_conv1')
            wk = tf.reshape(wk, [n, h*w, 1])

            wk_softmax = tf.nn.softmax(wk, 1)

            x_reshape = tf.transpose(x, [0,3,1,2])
            x_reshape = tf.reshape(x_reshape, [n, c, -1])
            f = tf.matmul(x_reshape, wk)
            y = tf.reshape(f, [n, 1, 1, out_dim])

            wv_1 = self.conv_layer(y, filters=out_dim/self.ratio, kernel=1, stride=1, layer_name=scope+'_conv2')
            wv_1 = self.Batch_Normalization(wv_1, training=self.training, scope=scope+'_batch1')
            wv_1 = self.Relu(wv_1)
            wv_2 = self.conv_layer(wv_1, filters=out_dim, kernel=1, stride=1, layer_name=scope+'_conv3')

            z = x + wv_2

            return z

    def Residual_block(self, x, out_dim, block_name, n_block):
        for i in range(n_block):
            input_x = x
            input_dim = int(np.shape(x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
            elif input_dim * 4 == out_dim:
                flag = True
                stride = 1
            else:
                flag = False
                stride = 1                

            x = self.transform_layer(input_x, stride=stride, out_dim=out_dim, scope='transform_layer_'+block_name+'_'+str(i))
            if self.mode == 'SENet':
                x = self.se_block(x, out_dim=out_dim, scope='se_block_'+block_name+'_'+str(i))
            if self.mode == 'NLNet':
                x = self.nl_block(x, out_dim=out_dim, scope='nl_block_'+block_name+'_'+str(i))
            if self.mode == 'SNLNet':
            x = self.snl_block(x, out_dim=out_dim, scope='snl_block_'+block_name+'_'+str(i))            
            if self.mode == 'GCNet':
            x = self.gc_block(x, out_dim=out_dim, scope='gc_block_'+block_name+'_'+str(i))

            if flag is True :
                branch_x = self.transistion_layer(input_x, stride=stride, out_dim=out_dim, scope='transistion_layer_'+block_name+'_'+str(i))
            else :
                branch_x = input_x

            x = self.Relu(x + branch_x)

        return x

    def Build_SEnet(self, x, training):
        self.training = training
        x = self.first_layer(x, scope='First_layer')
        #x = tf.layers.max_pooling2d(inputs=x, pool_size=3, strides=2, padding='SAME')

        x = self.Residual_block(x, out_dim=256, block_name='Residual_block_1', n_block=3)
        x = self.Residual_block(x, out_dim=512, block_name='Residual_block_2', n_block=4)
        x = self.Residual_block(x, out_dim=1024, block_name='Residual_block_3', n_block=6)
        x = self.Residual_block(x, out_dim=2048, block_name='Residual_block_4', n_block=3)

        x = self.Global_Average_Pooling(x)
        x = flatten(x)

        x = self.Fully_connected(x, units=10, layer_name='Fully_connected')

        return x

    def Evaluate(self, sess):
        test_acc = 0.0
        test_loss = 0.0
        test_pre_index = 0

        for it in range(1, int(10000/self.batch_size) + 1):
            test_batch_x = test_x[test_pre_index: test_pre_index + self.batch_size]
            test_batch_y = test_y[test_pre_index: test_pre_index + self.batch_size]
            test_pre_index = test_pre_index + self.batch_size

            test_feed_dict = {
                x: test_batch_x,
                label: test_batch_y,
                learning_rate: self.learning_rate,
                training_flag: False
            }

            loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

            test_loss += loss_
            test_acc += acc_

        test_loss /= (10000/self.batch_size)
        test_acc /= (10000/self.batch_size)

        return test_acc, test_loss

    def train(self):
        x = tf.placeholder(tf.float32, shape=[self.batch_size, 32, 32, 3])
        label = tf.placeholder(tf.float32, shape=[self.batch_size, 10])
        training_flag = tf.placeholder(tf.bool)
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        logits = self.Build_SEnet(x, training=training_flag)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))


        L2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=self.momentum, use_nesterov=True)
        train = optimizer.minimize(cost + L2_loss * self.weight_decay)
        train = optimizer.minimize(cost)

        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state('./model')
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            for epoch in range(1, self.total_epochs + 1):
                if epoch % 30 == 0 :
                    self.learning_rate = self.learning_rate / 10

                train_acc = 0.0
                train_loss = 0.0
                pre_index = 0

                train_x, train_y, test_x, test_y = prepare_data(self.dataset_path)
                train_x, test_x = color_preprocessing(train_x, test_x)

                for step in range(1, int(50000/self.batch_size) + 1):
                    batch_x = train_x[pre_index: pre_index + self.batch_size]
                    batch_y = train_y[pre_index: pre_index + self.batch_size]
                    pre_index += self.batch_size

                    batch_x = data_augmentation(batch_x)

                    train_feed_dict = {
                        x: batch_x,
                        label: batch_y,
                        learning_rate: self.learning_rate,
                        training_flag: True
                    }

                    _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
                    batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                    train_loss += batch_loss
                    train_acc += batch_acc
                    

                    line = "epoch: %d/%d, step: %d, batch_loss: %.4f, batch_acc: %.4f \n" % (
                    epoch, self.total_epochs, step, batch_loss, batch_acc)
                    print(line)

                train_loss /= (50000/self.batch_size)
                train_acc /= (50000/self.batch_size)

                test_acc, test_loss = self.Evaluate(sess)

                line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                    epoch, self.total_epochs, train_loss, train_acc, test_loss, test_acc)
                print(line)

                with open('logs.txt', 'a') as f:
                    f.write(line)

                saver.save(sess=sess, save_path='./model/SE_ResNet50.ckpt')