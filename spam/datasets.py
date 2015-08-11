import json
import random


def each_json(fn, max_count):
    print 'Pulling samples from file: %s' % fn
    i = 0
    with open(fn) as f:
        for line in f:
            if i == max_count:
                break
            j = json.loads(line)
            yield j
            i += 1
    assert i == max_count


def all_impermium_one_source(
        max_samples_per_class=45000, source='fyre'):
    pairs = []

    fn = '/media/data/knighton/data/spam_%s.txt' % source
    for j in each_json(fn, max_samples_per_class):
        pair = (j, 'spam')
        pairs.append(pair)

    fn = '/media/data/knighton/data/ham_%s.txt' % source
    for j in each_json(fn, max_samples_per_class):
        pair = (j, 'ham')
        pairs.append(pair)

    random.shuffle(pairs)

    n = len(pairs)
    train = pairs[:int(n * 0.8)]
    dev = pairs[int(n * 0.8):int(n * 0.9)]
    test = pairs[int(n * 0.9):]

    train = zip(*train)
    dev = zip(*dev)
    test = zip(*test)

    return train, dev, test
