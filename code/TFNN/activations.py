#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    激活函数
"""
import tensorflow as tf


def get_activation(activation=None):
    """
    Get activation function accord to the parameter 'activation'
    Args:
        activation: str: 激活函数的名称
    Return:
        激活函数
    """
    if activation is None:
        return None
    elif activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'relu':
        return tf.nn.relu
    elif activation == 'softmax':
        return tf.nn.softmax
    else:
        raise Exception('Unknow activation function: %s' % activation)
