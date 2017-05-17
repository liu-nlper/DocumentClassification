#!/usr/bin/env python
# coding=utf-8
import codecs
import pickle
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models.keyedvectors import KeyedVectors
from TFNN.utils.io_util import read_lines


def get_sentence(sentence_tag):
    words = []
    for item in sentence_tag.split(' '):
        index = item.rindex('/')
        words.append(item[:index])
    return ' '.join(words)


def extract_sentece():
    lines = read_lines('./Data/corpus/training.seg.csv')
    lines += read_lines('./Data/corpus/testing.seg.csv')
    with codecs.open('./Data/corpus/sentence.txt', 'w', encoding='utf-8') as file_w:
        for line in lines:
            index = line.index(',')
            word_tag = line[index+1:]
            file_w.write('%s\n' % get_sentence(word_tag))


def train():
    extract_sentece()

    in_path = './Data/corpus/sentence.txt'
    out_path = './Data/embedding/word2vec.bin'
    # 训练模型
    model = Word2Vec(
        sg=1, sentences=LineSentence(in_path),
        size=256, window=5, min_count=3, workers=4, iter=40)
    model.wv.save_word2vec_format(out_path, binary=True)


def bin2pkl():
    model = KeyedVectors.load_word2vec_format('./Data/embedding/word2vec.bin', binary=True)
    word_dict = {}
    for word in model.vocab:
        word_dict[word] = model[word]
    with open('./Data/embedding/word2vec.pkl', 'wb') as file_w:
        pickle.dump(word_dict, file_w)
        print(file_w.name)



if __name__ == '__main__':
    train()

    bin2pkl()
