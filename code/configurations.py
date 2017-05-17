#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    configurations
"""
import os


# --- corpus ---
TRAIN_PATH = './Data/corpus/training.seg.csv'
TEST_PATH = './Data/corpus/testing.seg.csv'


# --- voc ---
VOC_ROOT = './Data/voc'
if not os.path.exists(VOC_ROOT):
    os.mkdir(VOC_ROOT)
WORD_VOC_PATH = VOC_ROOT + '/word_voc.pkl'
WORD_VOC_START = 2
TAG_VOC_PATH = VOC_ROOT + '/tag_voc.pkl'
TAG_VOC_START = 1
LABEL_VOC_PATH = VOC_ROOT + '/label_voc.pkl'


# --- embedding ---
W2V_DIM = 256
W2V_PATH = './Data/embedding/word2vec.pkl'
EMBEDDING_ROOT = './Data/embedding/'
if not os.path.exists(EMBEDDING_ROOT):
    os.mkdir(EMBEDDING_ROOT)
W2V_TRAIN_PATH = EMBEDDING_ROOT + '/word2v.pkl'
T2V_PATH = EMBEDDING_ROOT + '/tag2v.pkl'
TAG_DIM = 64


# --- training param ---
MAX_LEN = 300
BATCH_SIZE = 64
NB_LABELS = 11
NB_EPOCH = 30
KEEP_PROB = 0.5
WORD_KEEP_PROB = 0.9
TAG_KEEP_PROB = 0.9
KFOLD = 10
