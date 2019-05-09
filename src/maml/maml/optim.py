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

"""Modified PyTorch Optimizer for retaining computation graphs

The general rule for porting a PyTorch optimizer is

    1. Inherit the PyTorch optimizer
    2. Override the ``step`` method by
        1. Add a ``retain_graph`` argument
        3. Rewrite step to
            1. Remove any in-place operations
            2. use a clone of the gradient for scaling factors that involve the gradient
            3. Return a list of all parameters

Expected behavior: if ``retain_graph=False``, revert to default behavior.
Else, use overriden method.

The overridden method will create a computational graph and return this
in the ``new_parameters`` list created through (c). Note that the
*optimizer* still keeps the original node, so after taking a ``step``,
it is necessary to replace the ``_parameters`` dict underlying
the model (assuming it inherits ``nn:Module``).
"""
import math
import torch
from torch import optim as _optim

# pylint: disable=too-many-arguments, invalid-name, redefined-builtin, protected-access, too-many-locals, arguments-differ, missing-docstring
# pylint: disable=too-few-public-methods


class SGD(_optim.SGD):


    def __init__(self, *args, detach=False, **kwargs):
        self.detach = detach
        super(SGD, self).__init__(*args, **kwargs)


    def step(self, closure=None, retain_graph=False):
        """Performs a single optimization step but retain the computational graph.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not retain_graph:
            return super(SGD, self).step(closure)

        loss = None
        if closure is not None:
            loss = closure()

        new_params = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            new_pg = []
            for p in group['params']:
                if p.grad is None:
                    new_params.append(p)
                    continue

                d_p = p.grad if not self.detach else p.grad.detach()

                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf = buf.mul(momentum).add(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf = buf.mul(momentum).add(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p + momentum * buf
                    else:
                        d_p = buf

                p = p - group['lr'] * d_p
                p.retain_grad()

                new_params.append(p)
                new_pg.append(p)
            group['params'] = new_pg
        return loss, new_params


class Adam(_optim.Adam):


    def __init__(self, *args, detach=False, **kwargs):
        self.detach = detach
        super(Adam, self).__init__(*args, **kwargs)


    def step(self, closure=None, retain_graph=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if not retain_graph:
            return super(Adam, self).step(closure)

        loss = None
        if closure is not None:
            loss = closure()

        new_params = []
        for group in self.param_groups:
            new_pg = []
            for p in group['params']:
                if p.grad is None:
                    new_params.append(p)
                    new_pg.append(p)
                    continue

                grad = p.grad if not self.detach else p.grad.detach()

                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead') # pylint: disable=line-too-long
                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p)

                # Decay the first and second moment running average coefficient
                g = grad.detach()
                exp_avg = exp_avg.mul(beta1).add(1 - beta1, grad)
                exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(1 - beta2, g, g)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    max_exp_avg_sq = torch.max(max_exp_avg_sq, exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p = p.addcdiv(-step_size, exp_avg, denom)
                p.retain_grad()

                new_params.append(p)
                new_pg.append(p)
            group['params'] = new_pg
        return loss, new_params
