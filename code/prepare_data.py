#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    prepare data.

    生成:
        word voc
        position voc
        relation type voc

        lookup tables
"""
import os
import pickle
import numpy as np
import configurations as config
from TFNN.utils.data_util import create_dictionary
from TFNN.utils.io_util import read_lines
from time import time


def init_voc():
    """
    初始化voc
    """
    lines = read_lines(config.TRAIN_PATH)
    lines += read_lines(config.TEST_PATH)
    words = []  # 句子
    pos_tags = []  # 词性标记类型
    for line in lines:
        index = line.index(',')
        sentence = line[index+1:]
        # words and tags
        words_tags = sentence.split(' ')
        words_temp, tag_temp = [], []
        for item in words_tags:
            r_index = item.rindex('/')
            word, tag = item[:r_index], item[r_index+1:]
            words_temp.append(word)
            tag_temp.append(tag)
        pos_tags.extend(tag_temp)
        words.extend(words_temp)
    # word voc
    create_dictionary(
        words, config.WORD_VOC_PATH, start=config.WORD_VOC_START,
        min_count=5, sort=True, lower=True, overwrite=True)
    # tag voc
    create_dictionary(
        pos_tags, config.TAG_VOC_PATH, start=config.TAG_VOC_START,
        sort=True, lower=False, overwrite=True)
    # label voc
    label_types = [str(i) for i in range(1, 12)]
    create_dictionary(
        label_types, config.LABEL_VOC_PATH, start=0, overwrite=True)


def init_word_embedding(path=None, overwrite=False):
    """
    初始化word embedding
    Args:
        path: 结果存放路径
    """
    if os.path.exists(path) and not overwrite:
        return
    with open(config.W2V_PATH, 'rb') as file:
        w2v_dict_full = pickle.load(file)
    with open(config.WORD_VOC_PATH, 'rb') as file:
        w2id_dict = pickle.load(file)
    word_voc_size = len(w2id_dict.keys()) + config.WORD_VOC_START
    word_weights = np.zeros((word_voc_size, config.W2V_DIM), dtype='float32')
    for word in w2id_dict:
        index = w2id_dict[word]  # 词的标号
        if word in w2v_dict_full:
            word_weights[index, :] = w2v_dict_full[word]
        else:
            random_vec = np.random.uniform(
                -0.25, 0.25, size=(config.W2V_DIM,)).astype('float32')
            word_weights[index, :] = random_vec
    # 写入pkl文件
    with open(path, 'wb') as file:
        pickle.dump(word_weights, file, protocol=2)


def init_tag_embedding(path, overwrite=False):
    """
    初始化pos tag embedding
    Args:
        path: 结果存放路径
    """
    if os.path.exists(path) and not overwrite:
        return
    with open(config.TAG_VOC_PATH, 'rb') as file:
        tag_voc = pickle.load(file)
    tag_voc_size = len(tag_voc.keys()) + config.TAG_VOC_START
    tag_weights = np.random.normal(
        size=(tag_voc_size, config.TAG_DIM)).astype('float32')
    for i in range(config.TAG_VOC_START):
        tag_weights[i, :] = 0.
    with open(path, 'wb') as file:
        pickle.dump(tag_weights, file, protocol=2)


def init_embedding():
    """
    初始化embedding
    """
    if not os.path.exists(config.EMBEDDING_ROOT):
        os.mkdir(config.EMBEDDING_ROOT)
    # 初始化word embedding
    init_word_embedding(config.W2V_TRAIN_PATH, overwrite=True)
    # 初始化tag embedding
    init_tag_embedding(config.T2V_PATH, overwrite=True)


def demo():
    with open(config.W2V_TRAIN_PATH, 'rb') as file:
        temp = pickle.load(file)
    print(temp.shape)


if __name__ == '__main__':
    t0 = time()

    init_voc()  # 初始化voc

    init_embedding()  # 初始化embedding

    demo()

    print('Done in %.1fs!' % (time()-t0))
