import os
import math
import torch
from torch import nn


class TimeMeter(object):
    def __init__(self, epoch):
        self.epoch = epoch
        self.fmt = ':0>2d'
        self.reset()

    def reset(self):
        self.time_start = 0
        self.time_end = 0

    def update_start(self, val):
        self.time_start = val

    def update_end(self, val):
        self.time_end = val

    def __str__(self):
        self.epoch -= 1
        eta = self.epoch * (self.time_end - self.time_start).seconds
        self.hours, rem = divmod(eta, 3600)
        self.minutes, self.seconds = divmod(rem, 60)
        fmtstr = 'eta {hours' + self.fmt + '}:{minutes' + self.fmt + '}:{seconds' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class MetricsMeter(object):
    def __init__(self, fmt=':f'):
        self.fmt = fmt

    def update(self, value):
        self.value = value

    def __str__(self):
        ls_str = [('{} {' + self.fmt + '}').format(k, v)
                  for (k, v) in self.value.items()]
        return ' '.join(ls_str)


class ProgressMeter(object):
    def __init__(self, num_epochs, meters, prefix=''):
        self.epoch_fmtstr = self._get_epoch_fmtstr(num_epochs)
        self.meters = meters
        self.prefix = prefix

    def _get_epoch_fmtstr(self, num_epochs):
        num_digits = len(str(num_epochs // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_epochs) + ']'

    def str(self, batch):
        entries = [self.prefix + self.epoch_fmtstr.format(batch)
                   ] + [str(meter) for meter in self.meters]
        txt = ' '.join(entries)
        return txt


def adjust_learning_rate(optimizer, init_lr, epoch, max_epoch):
    '''Decay the learning rate based on schedule'''
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def build_optimizer(model, init_lr, cfg):
    return torch.optim.SGD(
        model.parameters(),
        lr=init_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )


def build_criterion_aet():
    return nn.MSELoss().cuda()


def build_criterion_sim():
    return nn.CosineSimilarity().cuda()


def build_criterion_cls():
    return nn.CrossEntropyLoss().cuda()


def save_checkpoint(state, dirname, filename):
    torch.save(state, os.path.join(dirname, filename))


def log(log_file, str):
    with open(log_file, 'a') as fout:
        fout.write(str)


def calc_metrics(mat):
    num_class = len(mat)
    n = mat.sum(axis=1)
    N = mat.sum()
    TP = [mat[i, i] for i in range(num_class)]
    FP = [sum(mat[:, i]) - mat[i, i] for i in range(num_class)]
    FN = [sum(mat[i, :]) - mat[i, i] for i in range(num_class)]
    P = [TP[i] / (TP[i] + FP[i] + 1e-100) for i in range(num_class)]
    R = [TP[i] / (TP[i] + FN[i] + 1e-100) for i in range(num_class)]
    Pmi = sum(TP) / N
    F1 = 2 / N * sum(n[i] * P[i] * R[i] / (P[i] + R[i] + 1e-100)
                     for i in range(num_class))
    return Pmi, F1


def sanity_check(state_dict, checkpoint):
    print(f'==> Loading {checkpoint} for sanity check...')
    state_dict_pre = torch.load(checkpoint, map_location='cpu')

    for k in list(state_dict.keys()):
        assert ((state_dict[k].cpu() == state_dict_pre[k]).all()), \
            '==> {} is changed in classifier training.'.format(k)

    print('==> Sanity check passed.')
