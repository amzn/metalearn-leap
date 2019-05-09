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

from .utils import compute_global_norm


class Updater(object):

    """Gradient path update class.

    Arguments:
        state (OrderedDict): the model state dict to accumulate meta-gradients in.
        norm (bool): Use the gradient path length formula (d_1). If False, use the energy formula (d_2) (default=True).
        loss (bool): include the loss in the task manifold. If False, use only parameter distances (default=True).
        stabilizer (bool): use the stabilizer (mu) when loss=True (default=True).
    """

    def __init__(self, state, norm=True, loss=True, stabilizer=True):
        self.state = state
        self.norm = norm
        self.loss = loss
        self.stabilizer = stabilizer

        self._prev_loss = None
        self._prev_state = None

    def __call__(self, curr_loss, curr_state, *args, **kwargs):
        if self._prev_loss is not None:
            self.increment_grad(curr_loss, curr_state, *args, **kwargs)

        self._prev_loss = curr_loss
        self._prev_state = curr_state

    def initialize(self):
        """Reset meta counters"""
        self._prev_loss = None
        self._prev_state = None

    def increment_grad(self, curr_loss, curr_state, hook=None):
        """Accumulate gradients in a state dictionary

        Arguments:
            curr_loss (torch.Tensor): cloned loss at current iteration
            curr_state (OrderedDict): cloned state dict at current iteration
            hook (function): hook to process partial gradients with.
                A hook takes a torch.Tensor as input and makes in-place modifications (default=None).
        """
        prev_loss = self._prev_loss
        prev_state = self._prev_state
        d_loss = None
        norm = None

        if self.loss:
            d_loss = curr_loss - prev_loss
            if d_loss > 0 and self.stabilizer:
                d_loss = -d_loss

        if self.norm:
            norm = compute_global_norm(curr_state, prev_state, d_loss)

        for n, p in self.state.items():
            if not p.requires_grad:
                continue

            curr_param = curr_state[n].detach()
            prev_param = prev_state[n].detach()
            prev_param_grad = prev_state[n].grad.detach()

            add = prev_param - curr_param
            if self.loss:
                add += -d_loss * prev_param_grad

            if self.norm:
                add.data.div_(norm)

            if hook is not None:
                hook(add)

            p.grad.add_(add)
