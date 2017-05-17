#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Load data.
"""
import pickle
from time import time
import numpy as np
import configurations as config
from TFNN.utils.io_util import read_lines
from TFNN.utils.data_util import map_item2id


def get_sentence_arr(words_tags, word_voc, tag_voc):
    """
    获取词序列
    Args:
        words_tags: list, 句子 and tags
        word_voc: 词表
        tag_voc: 词性标注表
    Returns:
        sentence_arr: np.array, 字符id序列
        tag_arr: np.array, 词性标记序列
    """
    words, postags = [], []
    for item in words_tags:
        rindex = item.rindex('/')
        words.append(item[:rindex])
        postags.append(item[rindex+1:])
    # sentence arr
    sentence_arr = map_item2id(
        words, word_voc, config.MAX_LEN, lower=True)
    # pos tags arr
    postag_arr = map_item2id(
        postags, tag_voc, config.MAX_LEN, lower=False)
    return sentence_arr, postag_arr, len(words)


def init_data(lines, word_voc, tag_voc, label_voc):
    """
    加载数据
    Args:
        lines: list
        word_voc: dict, 词表
        tag_voc: dict, 词性标注表
        label_voc: dict
    Returns:
        sentences: np.array
        etc.
    """
    data_count = len(lines)
    sentences = np.zeros((data_count, config.MAX_LEN), dtype='int32')
    tags = np.zeros((data_count, config.MAX_LEN), dtype='int32')
    sentence_actual_lengths = np.zeros((data_count,), dtype='int32')
    labels = np.zeros((data_count,), dtype='int32')
    instance_index = 0
    for i in range(data_count):
        index = lines[i].index(',')
        label = lines[i][:index]
        sentence = lines[i][index+1:]
        words_tags = sentence.split(' ')
        sentence_arr, tag_arr, actual_length = get_sentence_arr(words_tags, word_voc, tag_voc)

        sentences[instance_index, :] = sentence_arr
        tags[instance_index, :] = tag_arr
        sentence_actual_lengths[instance_index] = actual_length
        labels[instance_index] = label_voc[label] if label in label_voc else 0
        instance_index += 1
    return sentences, tags, labels


def load_embedding():
    """
    加载词向量、词性向量
    Return:
        word_weights: np.array
        tag_weights: np.array
    """
    # 加载词向量
    with open(config.W2V_TRAIN_PATH, 'rb') as file_r:
        word_weights = pickle.load(file_r)
    # 加载tag向量
    with open(config.T2V_PATH, 'rb') as file_r:
        tag_weights = pickle.load(file_r)
    return word_weights, tag_weights


def load_voc():
    """
    Load voc...
    Return:
        word_voc: dict
        tag_voc: dict
        label_voc: dict
    """
    with open(config.WORD_VOC_PATH, 'rb') as file_r:
        word_voc = pickle.load(file_r)
    with open(config.TAG_VOC_PATH, 'rb') as file_r:
        tag_voc = pickle.load(file_r)
    with open(config.LABEL_VOC_PATH, 'rb') as file_r:
        label_voc = pickle.load(file_r)
    return word_voc, tag_voc, label_voc


def load_train_data(word_voc, tag_voc, label_voc):
    """
    加载训练测试数据
    Args:
        word_voc: dict
        tag_voc: dict
        label_voc: dict
    Returns:
        xx
    """
    return init_data(read_lines(config.TRAIN_PATH), word_voc, tag_voc, label_voc)


def load_test_data(word_voc, tag_voc, label_voc):
    """
    加载测试数据
    Args:
        word_voc: dict
        tag_voc: dict
        label_voc: dict
    Returns:
        xx
    """
    sentences, tags, _ = init_data(read_lines(config.TEST_PATH), word_voc, tag_voc, label_voc)
    return sentences, tags


def demo():
    t0 = time()
    word_weights, tag_weights = load_embedding()
    word_voc, tag_voc, label_voc = load_voc()
    data, label_voc = load_train_data()
    sentences, tags, labels = data[:]
    print(sentences.shape)
    print(tags.shape)
    print(labels.shape)
    print(word_weights.shape)
    print(tag_weights.shape)
    print('Done in %ds!' % (time()-t0))


if __name__ == '__main__':
    demo()
