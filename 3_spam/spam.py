#!/usr/bin/env python

from argparse import ArgumentParser
import datasets
from glob import glob
from keras_util import EarlyTermination, SaveModelsAndTerminateEarly
import networks
import numpy as np
import pipelines
import random


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--source', type=str, default='fyre')
    ap.add_argument('--samples_per_class', type=int, default=5000)
    ap.add_argument('--network', type=str, default='dense_0_256_6')
    return ap.parse_args()


def get_epoch(f):
    x = f.index('epoch_') + len('epoch_')
    e = f[x:]
    e = e[:e.index('_')]
    return int(e)


def get_last_checkpoint(d):
    ff = glob(d + '/*')
    if not ff:
        return None, 0

    ff.sort()
    f = ff[-1]

    return f, get_epoch(f) + 1


def transform_data((train, val, test), X_tf, y_tf):
    train = (X_tf.fit_transform(train[0]),
             y_tf.fit_transform(train[1]))
    val = (X_tf.transform(val[0]), y_tf.transform(val[1]))
    test = (X_tf.transform(test[0]), y_tf.transform(test[1]))
    return train, val, test


def build_model(data, network_name, start_weights):
    X, y = data[0]
    network, num_frontends = \
        networks.make_network(X, y, network_name)

    if start_weights:
        print 'Loading weights from', start_weights
        network.load_weights(start_weights)

    print 'Compiling'
    network.compile(loss='categorical_crossentropy',
                    optimizer='adam')

    return network, num_frontends


def train(data, model, num_frontends, resume_epoch, model_dir):
    cb = SaveModelsAndTerminateEarly()
    cb.set_params(model_dir, resume_epoch)
    X_train, y_train = data[0]
    X_val, y_val = data[1]

    if 1 < num_frontends:
        X_train = [X_train] * num_frontends
        X_val = [X_val] * num_frontends

    try:
        model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            nb_epoch=10000, batch_size=128, callbacks=[cb],
            show_accuracy=True)
    except EarlyTermination:
        pass


def run(args):
    random.seed(1337)
    np.random.seed(1337)

    print 'Loading raw data'
    raw_data = datasets.all_impermium_one_source(
        args.samples_per_class, args.source)

    print 'Building transformers'
    X_tf, y_tf = pipelines.json_to_ints2d()

    print 'Transforming data'
    data = transform_data(raw_data, X_tf, y_tf)

    print 'Locating the most recent checkpoint'
    model_dir = 'models/%d_%s' % (
        args.samples_per_class, args.network)
    start_weights, resume_epoch = get_last_checkpoint(model_dir)

    print 'Building model'
    network, num_frontends = \
        build_model(data, args.network, start_weights)

    print 'Training'
    train(data, network, num_frontends, resume_epoch, model_dir)


if __name__ == '__main__':
    run(parse_args())
