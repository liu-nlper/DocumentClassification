#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf


def zero_nil_slot(t, name=None):
    """
    Overwrite the nil_slot (first 1 rows) of the input Tensor with zeros.
    Args:
        t: 2D tensor
        name: str
    Returns:
        Same shape as t
    """
    with tf.name_scope('zero_nil_slot'):
        s = tf.shape(t)[1]
        z = tf.zeros([1, s], dtype=tf.float32)
        return tf.concat(
            axis=0, name=name,
            values=[z, tf.slice(t, [1, 0], [-1, -1])])


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    Args:
        t: 2D tensor
    Returns:
        2D tensor, same shape as t
    """
    with tf.name_scope("add_gradient_noise"):
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


def mask_tensor(input_data, lengths, maxlen, dtype=tf.float32):
    """
    Args:
        input_data: 2D tensor
        lengths: integer vector, all its values < maxlen
        maxlen: scalar integer tensor
        dtype: str
    """
    mask = tf.cast(tf.sequence_mask(lengths, maxlen), dtype)
    return tf.multiply(input_data, mask)
