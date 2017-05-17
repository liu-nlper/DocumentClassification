#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import codecs


def read_lines(path):
    lines = []
    with codecs.open(path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            line = line.rstrip()
            if line:
                lines.append(line)
    return lines


def get_file_list(path, postfix, file_list):
    """
    获取path路径下所有后缀为postfix的文件名
    Args:
        path str : 文件路径
        postfix str : 后缀
        file_list 存放文件路径
    Return:
        None
    """
    temp_list = os.listdir(path)
    for fi in temp_list:
        fi_d = os.path.join(path, fi)
        if os.path.isdir(fi_d):  # 若是目录，则递归
            get_file_list(fi_d, postfix, file_list)
        else:  # 若是文件
            if fi_d.endswith(postfix):  # 以postfix结尾
                file_list.append(fi_d)
    return None
