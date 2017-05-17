#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Layers:
    Convolutional1D

"""
import numpy as np
import tensorflow as tf
# from ..initializations import normal_weight
from ..activations import get_activation


class Convolutional1D(object):

    def __init__(self, input_data, filter_length, nb_filter, strides=[1, 1, 1, 1],
                 padding='VALID', activation='tanh', pooling=True,
                 name='Convolutional1D'):
        """1D卷积层
        Args:
            input_data: 3D tensor of shape=[batch_size, in_height, in_width]
                in_channels is set to 1 when use Convolutional1D.
            filter_length: int, 卷积核的长度，用于构造卷积核，在
                Convolutional1D中，卷积核shape=[filter_length, in_width, in_channels, nb_filters]
            nb_filter: int, 卷积核数量
            padding: 默认'VALID'，暂时不支持设成'SAME'
            pooling: bool, 是否池化
        """
        assert padding in ('VALID'), 'Unknow padding %s' % padding
        # assert padding in ('VALID', 'SAME'), 'Unknow padding %s' % padding

        in_height, in_width = map(int, input_data.get_shape()[1:])
        self._input_data = tf.expand_dims(input_data, -1)  # shape=[x, x, x, 1]
        self._filter_length = filter_length
        self._nb_filter = nb_filter
        self._strides = strides
        self._padding = padding
        self._activation = get_activation(activation)
        self._name = name
        self.pooling = pooling

        filter_length = self._filter_length
        nb_filter = self._nb_filter
        with tf.name_scope('%s_%d' % (name, filter_length)):
            if activation != 'relu':
                fan_in = filter_length * in_width
                fan_out = nb_filter * (in_width-filter_length+1)
                w_bound = np.sqrt(6. / (fan_in + fan_out))
                self.weights = tf.Variable(
                    tf.random_uniform(
                        minval=-w_bound, maxval=w_bound, dtype='float32',
                        shape=[filter_length, in_width, 1, nb_filter]),
                    name='conv_weight')
                tf.summary.histogram('weights', self.weights)
            else:  # init weight for relu
                w_values = tf.random_normal(
                    shape=[filter_length, in_width, 1, nb_filter]
                ) * tf.sqrt(2. / (filter_length * in_width * nb_filter))
                self.weights = tf.Variable(w_values, name='conv_weight')
            # bias
            self.biases = tf.Variable(
                tf.constant(0.1, shape=[nb_filter, ]),
                name='conv_bias')
            tf.summary.histogram('biases', self.biases)

        self.call()

    def call(self):
        # 卷积  if padding='VALID', then conv_output's shape=
        #   [batch_size, in_height-filter_length+1, 1, nb_filters]
        conv_output = tf.nn.conv2d(
            input=self._input_data,
            filter=self.weights,
            strides=self._strides,
            padding=self._padding)

        # output's shape=[batch_size, new_height, 1, nb_filters]
        linear_output = tf.nn.bias_add(conv_output, self.biases)
        act_output = (
            linear_output if self._activation is None
            else self._activation(linear_output))
        if self.pooling:
            # max pooling, shape=[?, nb_filter]
            self._output = tf.reduce_max(tf.squeeze(act_output, [2]), 1)
        else:
            self._output = tf.squeeze(act_output, axis=2)  # [?, n-w+1, nb_filter]

    @property
    def input_data(self):
        return self._input_data

    @property
    def output(self):
        return self._output

    @property
    def output_dim(self):
        return self._nb_filter

    def get_weights(self):
        return self.weights
