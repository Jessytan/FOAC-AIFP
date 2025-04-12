import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, f1_score

def cluster_acc(args,y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    presudo_l=[]
    id={}
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    s=np.unique(y_true)
    id_true=np.min(np.where(s>=args.num_labeled_classes))
    for i in range(len(s)):
        id={'id':i,'data':s[i]}
        presudo_l.append(id)
    D = max(len(s), 5)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        for j in range(len(np.unique(y_true))):
            if presudo_l[j]['data']==y_true[i]:
                idn = presudo_l[j]['id']
                break
        w[y_pred[i], idn] += 1
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T
    if len(s)<5: 
        ind_map = {i: presudo_l[j]['data'] for i, j in ind if j<len(s)}
        return sum([w[i, j] for i, j in ind if i<5 and j>=id_true and j<len(s) ]) * 1.0 / y_pred.size,ind_map
    else:
        ind_map = {i: presudo_l[j]['data'] for i, j in ind}
        return sum([w[i, j] for i, j in ind if i<5 and j>=id_true ]) * 1.0 / y_pred.size,ind_map

def calc(args,known, unknown):
    known=np.array(known)
    unknown=np.array(unknown)
    y_pred = np.append(np.where(known<[args.num_labeled_classes], 0, 1) , np.where(unknown>=[args.num_labeled_classes], 1, 0))
    y_true = np.append(np.zeros(len(known)),np.ones(len(unknown)) )
    # y_true= np.append(np.where(np.isin(known, args.known), 0, 1) , np.where(np.isin(unknown, args.unknown), 1, 0))
    # y_pred = np.append(np.zeros(len(known)),np.ones(len(unknown)) )
    f_score = f1_score(y_true, y_pred, average="binary")
    return f_score

def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return round(self.v, 5)
    

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
