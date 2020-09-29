"""
Utilities
"""

from collections import Counter, OrderedDict
import json
import os
import shutil

import numpy as np
import torch

random_counter = [0]


def next_random():
    random = np.random.RandomState(random_counter[0])
    random_counter[0] += 1
    return random


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self), )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, raw=False):
        self.raw = raw
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.raw:
            self.raw_scores = []

    def update(self, val, n=1, raw_scores=None):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if self.raw:
            self.raw_scores.extend(list(raw_scores))


def save_checkpoint(state, is_best, folder='./',
                    filename='checkpoint.pth.tar'):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(folder, filename),
                        os.path.join(folder, 'model_best.pth.tar'))


def merge_args_with_dict(args, dic):
    for k, v in list(dic.items()):
        setattr(args, k, v)


def make_output_and_sample_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sample_dir = os.path.join(out_dir, 'samples')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    return out_dir, sample_dir


def save_defaultdict_to_fs(d, out_path):
    d = dict(d)
    with open(out_path, 'w') as fp:
        d_str = json.dumps(d, ensure_ascii=True)
        fp.write(d_str)


def idx2word(idx, i2w):
    sent_str = [str()] * len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            sent_str[i] += str(i2w[word_id.item()]) + " "
        sent_str[i] = sent_str[i].strip()

    return sent_str


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.optimize import linear_sum_assignment

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def featurize():
    image_model.eval()
    N_FEATS = final_feat_dim
    # DATA_DIR = '/Users/wangchong/Downloads/hard_sw'
    DATA_DIR = args.data_dir

    if preprocess:
        preprocess_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

    with torch.no_grad():
        for split in ("train", "val", "test", "val_same", "test_same"):
            print(split);
            data_loader = data_loader_dict[split]

            ex = np.load("{}/shapeworld/{}/examples.npz".format(DATA_DIR, split))['arr_0']
            ex = np.transpose(ex, (0, 1, 4, 2, 3))
            n_inp = ex.shape[0]
            n_ex = ex.shape[1]
            ex_feats = np.zeros((n_inp, n_ex, N_FEATS))
            for i in range(0, n_inp, args.batch_size):
                if i % 200 == 0:
                    print(i);
                batch = ex[i:i+args.batch_size, ...]
                n_batch = batch.shape[0]
                batch = torch.from_numpy(batch).float().to(device)
                if preprocess:
                    batch = batch.reshape(n_batch*n_ex, batch.shape[2], batch.shape[3], batch.shape[4]).cpu();
                    batch = torch.stack([preprocess_transform(b) for b in batch]).to(device)
                    batch = batch.reshape(n_batch, n_ex, batch.shape[1], 224, 224);
                feats = image_model(batch).cpu().numpy();
                ex_feats[i:i+args.batch_size, ...] = feats
            np.savez("{}/shapeworld/{}/examples.feats.npz".format(DATA_DIR, split), ex_feats);

            inp = np.load("{}/shapeworld/{}/inputs.npz".format(DATA_DIR, split))['arr_0']
            inp = np.transpose(inp, (0, 3, 1, 2))
            n_inp = inp.shape[0]
            inp_feats = np.zeros((n_inp, N_FEATS))
            for i in range(0, n_inp, args.batch_size):
                if i % 200 == 0:
                    print(i)
                batch = inp[i:i+args.batch_size, ...]
                batch = torch.from_numpy(batch).float().cpu()
                if preprocess:
                    batch = torch.stack([preprocess_transform(b) for b in batch]).to(device);
                feats = image_model(batch).cpu().numpy()
                feats = feats.reshape((-1, N_FEATS))
                inp_feats[i:i+args.batch_size, :] = feats
            np.savez("{}/shapeworld/{}/inputs.feats.npz".format(DATA_DIR, split), inp_feats)