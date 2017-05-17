#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
import codecs


def sim_compute(pro_labels, right_labels, ignore_label=None):
    """
    simple evaluate...
    Args:
        param pro_labels list : predict labels
        param right_labels list : right labels
        param ignore_label int : the label should be ignored
    Returns:
        pre, rec, f
    """
    assert len(pro_labels) == len(right_labels)
    pre_pro_labels, pre_right_labels = [], []
    rec_pro_labels, rec_right_labels = [], []
    labels_len = len(pro_labels)
    for i in range(labels_len):
        pro_label = pro_labels[i]
        if pro_label != ignore_label:  #
            pre_pro_labels.append(pro_label)
            pre_right_labels.append(right_labels[i])
        if right_labels[i] != ignore_label:
            rec_pro_labels.append(pro_label)
            rec_right_labels.append(right_labels[i])
    pre_pro_labels, pre_right_labels = np.array(pre_pro_labels, dtype='int32'), \
        np.array(pre_right_labels, dtype='int32')
    rec_pro_labels, rec_right_labels = np.array(rec_pro_labels, dtype='int32'), \
        np.array(rec_right_labels, dtype='int32')
    pre = 0. if len(pre_pro_labels) == 0 \
        else len(np.where(pre_pro_labels == pre_right_labels)[0]) / float(len(pre_pro_labels))
    # rec = len(np.where(rec_pro_labels == rec_right_labels)[0]) / float(len(pre_pro_labels))
    rec = len(np.where(rec_pro_labels == rec_right_labels)[0]) / float(len(rec_right_labels))
    f = 0. if (pre + rec) == 0. \
        else (pre * rec * 2.) / (pre + rec)
    return pre, rec, f


def demo():
    pro_labels = [1, 2, 3, 4, 0, 6, 7, 0, 2, 8]
    right_labels = [0, 2, 3, 6, 5, 4, 7, 1, 0, 3]
    # ignore_label = 0
    pre, rec, f = sim_compute(pro_labels, right_labels, ignore_label=2)
    print('pre:', pre)
    print('rec:', rec)
    print('  f:', f)


if __name__ == '__main__':
    demo()
