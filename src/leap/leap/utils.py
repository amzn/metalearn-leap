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

"""Transferring Knowledge across Learning Processes

Original implementation of the Leap algorithm, https://arxiv.org/abs/1812.01054.

"""
from collections import OrderedDict
import torch


def clone(tensor):
    """Detach and clone a tensor including the ``requires_grad`` attribute.

    Arguments:
        tensor (torch.Tensor): tensor to clone.
    """
    cloned = tensor.detach().clone()
    cloned.requires_grad = tensor.requires_grad
    if tensor.grad is not None:
        cloned.grad = clone(tensor.grad)
    return cloned


def clone_state_dict(state_dict):
    """Clone a state_dict. If state_dict is from a ``torch.nn.Module``, use ``keep_vars=True``.

    Arguments:
        state_dict (OrderedDict): the state_dict to clone. Assumes state_dict is not detached from model state.
    """
    return OrderedDict([(name, clone(param)) for name, param in state_dict.items()])


def compute_global_norm(curr_state, prev_state, d_loss):
    """Compute the norm of the line segment between current parameters and previous parameters.

    Arguments:
        curr_state (OrderedDict): the state dict at current iteration.
        prev_state (OrderedDict): the state dict at previous iteration.
        d_loss (torch.Tensor, float): the loss delta between current at previous iteration (optional).
    """
    norm = d_loss * d_loss if d_loss is not None else 0

    for name, curr_param in curr_state.items():
        if not curr_param.requires_grad:
            continue

        curr_param = curr_param.detach()
        prev_param = prev_state[name].detach()
        param_delta = curr_param.data.view(-1) - prev_param.data.view(-1)
        norm += torch.dot(param_delta, param_delta)
    norm = norm.sqrt()
    return norm
