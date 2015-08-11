from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout, Flatten, \
    TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
import numpy as np


def make_dense(X, y, num_layers, width, dropout):
    assert len(X.shape) == 2
    assert len(y.shape) == 2

    vocab_size = np.amax(X) + 1

    print 'Vocab size:', vocab_size

    m = Sequential()
    m.add(Embedding(vocab_size, 8))
    m.add(Dropout(dropout))

    m.add(TimeDistributedDense(8, 64))
    m.add(Flatten())

    m.add(BatchNormalization((64 * X.shape[1],)))
    m.add(PReLU((64 * X.shape[1],)))
    m.add(Dropout(dropout))
    m.add(Dense(64 * X.shape[1], width))

    for i in range(num_layers):
        m.add(BatchNormalization((width,)))
        m.add(PReLU((width,)))
        m.add(Dropout(dropout))
        m.add(Dense(width, width))

    m.add(BatchNormalization((width,)))
    m.add(PReLU((width,)))
    m.add(Dropout(dropout))
    m.add(Dense(width, y.shape[1]))

    m.add(Activation('softmax'))
    return m, 1


def make_network(X, y, name):
    ss = name.split('_')
    kind = ss[0]
    if kind == 'dense':
        num_layers, width, dropout = map(int, ss[1:])
        return make_dense(X, y, num_layers, width, dropout)
    else:
        assert False
