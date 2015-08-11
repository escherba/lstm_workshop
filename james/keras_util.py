from keras.callbacks import Callback
import os


def show_accuracy(f):
    if f == 1.0:
        f = 0.9999999999

    try:
        assert 0 <= f < 1
        s = '%.5f' % f
        return s[2:]
    except:
        return 'xxxx'


class EarlyTermination(Exception):
    pass


def is_hopeless(accs):
    if len(accs) < 5:
        return False

    for i in range(len(accs) - 5, len(accs) - 1):
        a = accs[i]
        b = accs[i + 1]
        if a < b:
            return False

    return True


class SaveModelsAndTerminateEarly(Callback):
    def on_epoch_begin(self, _1, _2):
        self.batch_accuracies = []
        self.epoch_accuracies = []

    def set_params(self, model_save_dir, resume_epoch=None):
        self.model_save_dir = model_save_dir
        if resume_epoch:
            self.epoch_offset = resume_epoch
        else:
            self.epoch_offset = 0

        self.acc_to_beat = 0.0
        self.epochs_left = None

    def on_batch_end(self, batch_index, logs):
        acc = logs['acc']
        self.batch_accuracies.append(acc)

    def maybe_terminate_early(self, val_acc):
        self.epoch_accuracies.append(val_acc)
        if is_hopeless(self.epoch_accuracies):
            raise EarlyTermination

    def on_epoch_end(self, epoch_index, logs):
        train_acc = sum(self.batch_accuracies) / len(self.batch_accuracies)
        train_s = show_accuracy(train_acc)

        val_acc = logs['val_acc']
        val_s = show_accuracy(val_acc)

        e = epoch_index + self.epoch_offset

        f = ('%s/epoch_%04d_train_%s_val_%s.h5' %
             (self.model_save_dir, e, train_s, val_s))
        d = os.path.dirname(f)
        print f, d
        if not os.path.isdir(d):
            print 'Creating...'
            os.makedirs(d)
        self.model.save_weights(f)

        self.maybe_terminate_early(val_acc)
