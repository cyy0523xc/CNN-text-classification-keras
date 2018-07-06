# -*- coding: utf-8 -*-
#
#
# Author: alex
# Created Time: 2018年07月06日 星期五 21时04分31秒
import numpy as np
np.random.seed(1337)  # for reproducibility

import jieba
from fire import Fire
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM


def load_data_and_labels(pos_file, neg_file, train_rate=0.8):
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    with open(pos_file, encoding='utf8') as r:
        positive_examples = list(r.readlines())

    with open(neg_file, encoding='utf8') as r:
        negative_examples = list(r.readlines())

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]

    train_x = positive_examples
    train_y = [1] * len(train_x)
    train_len = int(len(train_x)*train_rate)
    test_x = train_x[train_len:]
    test_y = train_y[train_len:]
    train_x = train_x[:train_len]
    train_y = train_y[:train_len]

    train_x2 = negative_examples
    train_y2 = [0] * len(train_x2)
    train_len = int(len(train_x2)*train_rate)
    train_x += train_x2[:train_len]
    train_y += train_y2[:train_len]
    test_x += train_x2[train_len:]
    test_y += train_y2[train_len:]

    train_x = [jieba.lcut(s) for s in train_x]
    test_x = [jieba.lcut(s) for s in test_x]
    return (train_x, train_y, test_x, test_y)


def train(pos_file, neg_file, train_rate=0.8, epoch=20, batch_size=32,
          maxlen=80, max_features=20000, in_dim=128, out_dim=128,
          loss='binary_crossentropy', optimizer='adam'
          ):
    """
    # 设定参数
    max_features = 20000   # 词汇表大小
    # cut texts after this number of words (among top max_features most common words)
    # 裁剪文本为 maxlen 大小的长度（取最后部分，基于前 max_features 个常用词）
    maxlen = 80
    batch_size = 32   # 批数据量大小
    """
    # 载入数据
    print('Loading data...')
    X_train, y_train, X_test, y_test = load_data_and_labels(pos_file, neg_file, train_rate)
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    # 裁剪为 maxlen 长度
    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    # 构建模型
    print('Build model...')
    model = Sequential()
    # 嵌入层，每个词维度为128
    model.add(Embedding(max_features, in_dim, dropout=0.2))
    # LSTM层，输出维度128，可以尝试着换成 GRU 试试
    model.add(LSTM(out_dim, dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
    model.add(Dense(1))   # 单神经元全连接层
    model.add(Activation('sigmoid'))   # sigmoid 激活函数层

    model.summary()   # 模型概述

    # try using different optimizers and different optimizer configs
    # 这里可以尝试使用不同的损失函数和优化器
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # 训练迭代，使用测试集做验证（真正实验时最好不要这样做）
    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epoch,
            validation_data=(X_test, y_test))

    # 评估误差和准确率
    score, acc = model.evaluate(X_test, y_test,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == '__main__':
    Fire(train)
