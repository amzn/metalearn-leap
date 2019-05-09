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

"""Utilities for constructing MAML objects"""
from collections import OrderedDict
import numpy as np

# pylint: disable=too-many-arguments, invalid-name, redefined-builtin, protected-access, too-many-locals


def compute_auc(x):
    """Compute AUC (composite trapezoidal rule)"""
    T = len(x)
    v = 0
    for i in range(1, T):
        v += ((x[i] - x[i-1]) / 2 + x[i-1]) / T
    return v


def n_correct(p, y):
    """Number correct predictions"""
    _, p = p.max(1)
    correct = (p == y).sum().item()
    return correct


def build_dict(names, parameters):
    """Populate an ordered dictionary of parameters"""
    state_dict = OrderedDict({n: p for n, p in zip(names, parameters)})
    return state_dict


def build_iterator(tensors, inner_bsz, outer_bsz, inner_steps, outer_steps, cuda=False, device=0):
    """Construct a task iterator from input and output tensor"""
    inner_size = inner_bsz * inner_steps
    outer_size = outer_bsz * outer_steps
    tsz = tensors[0].size(0)
    if tsz != inner_size + outer_size:
        raise ValueError(
            'tensor size mismatch: expected {}, got {}'.format(
                inner_size + outer_size, tsz))

    def iterator(start, stop, size):  #pylint: disable=missing-docstring
        for i in range(start, stop, size):
            out = tuple(t[i:i+size] for t in tensors)
            if cuda:
                out = tuple(t.cuda(device) for t in out)
            yield out

    return iterator(0, inner_size, inner_bsz), iterator(inner_size, tsz, outer_bsz)

###############################################################################


def _load_from_par_dict(module, par_dict, prefix):
    """Replace the module's _parameter dict with par_dict"""
    _new_parameters = OrderedDict()
    for name, param in module._parameters.items():
        key = prefix + name
        if key in par_dict:
            input_param = par_dict[key]
        else:
            input_param = param

        if input_param.shape != param.shape:
            # local shape should match the one in checkpoint
            raise ValueError(
                'size mismatch for {}: copying a param of {} from checkpoint, '
                'where the shape is {} in current model.'.format(
                    key, param.shape, input_param.shape))

        _new_parameters[name] = input_param
    module._parameters = _new_parameters


def load_state_dict(module, state_dict):
    r"""Replaces parameters and buffers from :attr:`state_dict` into
    the given module and its descendants. In contrast to the module's
    method, this function will *not* do in-place copy of underlying data on
    *parameters*, but instead replace the ``_parameter`` dict in each
    module and its descendants. This allows us to backpropr through previous
    gradient steps using the standard top-level API.

    .. note::
        You must store the original state dict (with keep_vars=True) separately
        and, when ready to update them, use :funct:`load_state_dict` to return
        as the module's parameters.

    Arguments:
        module (torch.nn.Module): a module instance whose state to update.
        state_dict (dict): a dict containing parameters and
            persistent buffers.
    """
    par_names = [n for n, _ in module.named_parameters()]

    par_dict = OrderedDict({k: v for k, v in state_dict.items() if k in par_names})
    no_par_dict = OrderedDict({k: v for k, v in state_dict.items() if k not in par_names})
    excess = [k for k in state_dict.keys()
              if k not in list(no_par_dict.keys()) + list(par_dict.keys())]

    if excess:
        raise ValueError(
            "State variables %r not in the module's state dict %r" % (excess, par_names))

    metadata = getattr(state_dict, '_metadata', None)
    if metadata is not None:
        par_dict._metadata = metadata
        no_par_dict._metadata = metadata

    module.load_state_dict(no_par_dict, strict=False)

    def load(module, prefix=''): # pylint: disable=missing-docstring
        _load_from_par_dict(module, par_dict, prefix)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)


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
        ncorr = n_correct(pred.data, target.data)
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
