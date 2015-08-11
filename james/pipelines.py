from camacho.base import Transformer
from camacho.pipelines import TransformerPipeline
from camacho.preprocess.sequence.coders import IntCoder
from camacho.preprocess.binarize.onehot import AtomBinarizer


class ExtractFrontBackText(Transformer):
    def __init__(self, length=128):
        self.length = length

    def transform(self, jj):
        n = self.length / 2
        ss = []
        for j in jj:
            s = j['object']['content']
            front = s[:n]
            if len(front) < n:
                front += '\0' * (n - len(front))
            back = s[-n:]
            if len(back) < n:
                back = '\0' * (n - len(back)) + back
            if self.length % 2:
                s = front + '\0' + back
            else:
                s = front + '\0' + back[1:]
            ss.append(s)
        return ss


def json_to_ints2d():
    X = TransformerPipeline([
        ExtractFrontBackText(length=128),
        IntCoder(min_freq=10),
    ])

    y = TransformerPipeline([
        AtomBinarizer(),
    ])

    return X, y
