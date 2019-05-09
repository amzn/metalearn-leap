# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

"""Runtime helpers"""
# pylint: disable=invalid-name,too-many-arguments,too-many-instance-attributes
import os
from os.path import join
import numpy as np


def convert_arg(arg):
    """Convert string to type"""
    # pylint: disable=broad-except
    if arg.lower() == 'none':
        arg = None
    elif arg.lower() == 'false':
        arg = False
    elif arg.lower() == 'true':
        arg = True
    elif '.' in arg:
        try:
            arg = float(arg)
        except Exception:
            pass
    else:
        try:
            arg = int(arg)
        except Exception:
            pass
    return arg


def build_kwargs(args):
    """Build a kwargs dict from a list of key-value pairs"""
    kwargs = {}

    if not args:
        return kwargs

    assert len(args) % 2 == 0, "argument list %r does not appear to have key, value pairs" % args

    while args:
        k = args.pop(0)
        v = args.pop(0)
        if ':' in v:
            v = tuple(convert_arg(a) for a in v.split(':'))
        else:
            v = convert_arg(v)
        kwargs[str(k)] = v

    return kwargs


def compute_ncorrect(p, y):
    """Accuracy over a tensor of predictions"""
    _, p = p.max(1)
    correct = (p == y).sum().item()
    return correct


def compute_auc(x):
    """Compute AUC (composite trapezoidal rule)"""
    T = len(x)
    v = 0
    for i in range(1, T):
        v += ((x[i] - x[i-1]) / 2 + x[i-1]) / T
    return v


def unlink(path):
    """Unlink logfiles"""
    for f in os.listdir(path):
        f = os.path.join(path, f)
        if f.endswith('.log'):
            os.unlink(f)

#################################################################################


def write(step, meta_loss, loss, accuracy, losses, accuracies, f):
    """Write results data to file"""
    lstr = ""
    for l in losses:
        lstr += "{:f};".format(l)

    astr = ""
    for a in accuracies:
        astr += "{:f};".format(a)

    msg = "{:d},{:f},{:f},{:f},{:s},{:s}\n".format(
        step, meta_loss, loss, accuracy, lstr, astr)

    with open(f, 'a') as fo:
        fo.write(msg)



def log_status(results, idx, time):
    """Print status"""
    #pylint: disable=unbalanced-tuple-unpacking,too-many-star-expressions
    print("[{:9s}] time:{:3.3f} "
          "train: outer={:0.4f} inner={:0.4f} acc={:2.2f} "
          "val: outer={:0.4f} inner={:0.4f} acc={:2.2f}".format(
              str(idx),
              time,
              results.train_meta_loss,
              results.train_loss,
              results.train_acc,
              results.val_meta_loss,
              results.val_loss,
              results.val_acc)
          )

def write_train_res(results, step, log_dir):
    """Write results from a meta-train step to file"""
    write(step,
          results.train_meta_loss,
          results.train_loss,
          results.train_acc,
          results.train_losses,
          results.train_accs,
          join(log_dir, 'results_train_train.log'))
    write(step,
          results.val_meta_loss,
          results.val_loss,
          results.val_acc,
          results.val_losses,
          results.val_accs,
          join(log_dir, 'results_train_val.log'))


def write_val_res(results, step, case, log_dir):
    """Write task results data to file"""
    for task_id, res in enumerate(results):
        write(step,
              res.train_meta_loss,
              res.train_loss,
              res.train_acc,
              res.train_losses,
              res.train_accs,
              join(log_dir, 'results_{}_{}_train.log'.format(task_id, case)))
        write(step,
              res.val_meta_loss,
              res.val_loss,
              res.val_acc,
              res.val_losses,
              res.val_accs,
              join(log_dir, 'results_{}_{}_val.log'.format(task_id, case)))

#################################################################################


class Res:

    """Results container
    Attributes:
        losses (list): list of losses over batch iterator
        accs (list): list of accs over batch iterator
        meta_loss (float): auc over losses
        loss (float): mean loss over losses. Call ``aggregate`` to compute.
        acc (float): mean acc over accs. Call ``aggregate`` to compute.
    """

    def __init__(self):
        self.losses = []
        self.accs = []
        self.ncorrects = []
        self.nsamples = []
        self.meta_loss = 0
        self.loss = 0
        self.acc = 0

    def log(self, loss, pred, target):
        """Log loss and accuracies"""
        nsamples = target.size(0)
        ncorr = compute_ncorrect(pred.data, target.data)
        accuracy = ncorr / target.size(0)

        self.losses.append(loss)
        self.ncorrects.append(ncorr)
        self.nsamples.append(nsamples)
        self.accs.append(accuracy)

    def aggregate(self):
        """Compute aggregate statistics"""
        self.accs = np.array(self.accs)
        self.losses = np.array(self.losses)
        self.nsamples = np.array(self.nsamples)
        self.ncorrects = np.array(self.ncorrects)

        self.loss = self.losses.mean()
        self.meta_loss = compute_auc(self.losses)
        self.acc = self.ncorrects.sum() / self.nsamples.sum()


class AggRes:

    """Results aggregation container
    Aggregates results over a mini-batch of tasks
    """

    def __init__(self, results):
        self.train_res, self.val_res = zip(*results)
        self.aggregate_train()
        self.aggregate_val()

    def aggregate_train(self):
        """Aggregate train results"""
        (self.train_meta_loss,
         self.train_loss,
         self.train_acc,
         self.train_losses,
         self.train_accs) = self.aggregate(self.train_res)

    def aggregate_val(self):
        """Aggregate val results"""
        (self.val_meta_loss,
         self.val_loss,
         self.val_acc,
         self.val_losses,
         self.val_accs) = self.aggregate(self.val_res)

    @staticmethod
    def aggregate(results):
        """Aggregate losses and accs across Res instances"""
        agg_losses = np.stack([res.losses for res in results], axis=1)
        agg_ncorrects = np.stack([res.ncorrects for res in results], axis=1)
        agg_nsamples = np.stack([res.nsamples for res in results], axis=1)

        mean_loss = agg_losses.mean()
        mean_losses = agg_losses.mean(axis=1)
        mean_meta_loss = compute_auc(mean_losses)

        mean_acc = agg_ncorrects.sum() / agg_nsamples.sum()
        mean_accs = agg_ncorrects.sum(axis=1) / agg_nsamples.sum(axis=1)

        return mean_meta_loss, mean_loss, mean_acc, mean_losses, mean_accs


def consolidate(agg_res):
    """Merge a list of agg_res into one agg_res"""
    results = [sum((r.train_res, r.val_res), ()) for r in agg_res]
    return AggRes(results)