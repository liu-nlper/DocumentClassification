#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf


def categorical_crossentropy(y_true, y_pred):
    """
    Args:
        y_true: int of list, length=batch_size
        y_pred: 2D tensor with shape=[batch_size, nb_classes]
    Returns:
        xx
    """
    cross_entroy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y_pred, labels=y_true, name='xentroy')
    return tf.reduce_mean(cross_entroy, name='xentroy_mean')
