#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np
from collections import defaultdict


def flatten_list(nest_list):
    """
    将嵌套列表压扁
    Args:
        nest_list: list,嵌套列表
    Return:
        flatten_list: list
    """
    res = []
    for item in nest_list:
        if isinstance(item, list):
            res.extend(flatten_list(item))
        else:
            res.append(item)
    return res


def create_dictionary(items, dic_path, start=0, sort=True,
                      min_count=None, lower=False, overwrite=False):
    """
    构建字典，并将构建的字典写入pkl文件中
    Args:
        items: list, [item_1, item_2, ...]
        dic_path: 需要保存的路径(以pkl结尾)
        start: int, voc起始下标，默认为0
        sort: bool, 是否按频率排序, 若为False，则按items排序
        min_count: 最小频次
        lower: bool, 是否转为小写
        overwrite: bool, 是否覆盖之前的文件
    Returns:
        None
    """
    assert not dic_path.endswith('pk')
    if os.path.exists(dic_path) and not overwrite:
        return
    voc = dict()
    if sort:
        # 构建字典
        dic = defaultdict(int)
        for item in items:
            item = item if (not lower) else item.lower()
            dic[item] += 1
        # 排序
        dic = sorted(dic.items(), key=lambda d: d[1], reverse=True)
        for i, item in enumerate(dic):
            index = i + start
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            voc[key] = index
    else:  # 按items排序
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            index = i + start
            voc[item] = index
    # 写入文件
    file = open(dic_path, 'wb')
    pickle.dump(voc, file)
    file.close()


def map_item2id(items, voc, max_len, none_word=0, lower=False):
    """
    将word/pos等映射为id
    Args:
        items: list, 待映射列表
        voc: 词表
        max_len: int, 序列最大长度
        none_word: 未登录词标号,默认为0
    Returns:
        arr: np.array, dtype=int32, shape=[max_len,]
    """
    assert type(none_word) == int
    arr = np.zeros((max_len,), dtype='int32')
    min_range = min(max_len, len(items))
    for i in range(min_range):  # 若items长度大于max_len，则被截断
        item = items[i] if not lower else items[i].lower()
        arr[i] = voc[item] if item in voc else none_word
    return arr


def random_over_sampling():
    """
    随机过采样
    Args:
        xx
    Return:
        xx
    """
    x_1 = [[1,1,1], [2,2,2], [3,3,3]]
    x_2 = [[1,1,1], [2,2,2], [3,3,3]]
    y = [1,2,3]
    from imblearn.over_samping import RandomOverSampler
    ros = RandomOverSampler(sandom_state=42)
    x_res, y_res = ros.fit_sample(x_1, y)
    print(x_res)
    print(y_res)


def demo():
    random_over_sampling()


if __name__ == '__main__':
    demo()
