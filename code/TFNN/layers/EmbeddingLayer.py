#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Layers:
    Embedding

"""
import tensorflow as tf


class Embedding(object):

    def __init__(self, params, ids, name, keep_prob=1.0):
        with tf.name_scope('%s' % name):
            self._params = tf.Variable(params, tf.float32, name='embed')
            self._ids = ids

            # output
            embed_output = tf.nn.embedding_lookup(
                params=self._params,
                ids=self._ids
            )
            self._output = tf.nn.dropout(embed_output, keep_prob)

    @property
    def params(self):
        return self._params

    @property
    def output_dim(self):
        return int(self._output.get_shape()[-1])

    @property
    def output(self):
        return self._output