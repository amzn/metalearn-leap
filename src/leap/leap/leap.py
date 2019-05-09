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
from .updaters import Updater
from .utils import clone_state_dict


class Leap(object):

    """Leap

    Meta-learner class used during meta-training to learn a gradient path distance minimizing initialization.

    Arguments:
        model (torch.nn.Module): The model whose parameters Leap should meta-learn.
        norm (bool): Use the gradient path length formula (d_1). If False, use the energy formula (d_2) (default=True).
        loss (bool): include the loss in the task manifold. If False, use only parameter distances (default=True).
        stabilizer (bool): use the stabilizer (mu) when loss=True (default=True).

    Example:
        >>> Require: criterion, model, tasks, opt_cls, meta_opt_cls, opt_kwargs, meta_opt_kwargs
        >>>
        >>> leap = Leap(model)
        >>> mopt = meta_opt_cls(leap.parameters(), **meta_opt_kwargs)
        >>> for meta_steps:
        >>>     meta_batch = tasks.sample()
        >>>     for task in meta_batch:
        >>>         leap.init_task()
        >>>         leap.to(model)
        >>>
        >>>         opt = opt_cls(model.parameters(), **opt_kwargs)
        >>>
        >>>         for x, y in task:
        >>>             loss = criterion(model(x), y)
        >>>             loss.backward()
        >>>
        >>>             leap.update(loss, model)
        >>>
        >>>             opt.step()
        >>>             opt.zero_grad()  # MUST come after leap.update
        >>>     ###
        >>>     leap.normalize()
        >>>     meta_optimizer.step()
        >>>     meta_optimizer.zero_grad()
    """

    def __init__(self, model, norm=True, loss=True, stabilizer=True):
        self.norm = norm
        self.loss = loss
        self.stabilizer = stabilizer

        self.state = clone_state_dict(model.state_dict(keep_vars=True))
        self.updater = Updater(self.state, self.norm, self.loss, self.stabilizer)
        self.zero()

        self._task_counter = 0

    def zero(self):
        """Zero out gradients tensors in the Leap state dict."""
        for param in self.parameters():
            if param.grad is None:
                param.grad = param.new(*param.shape)
            param.grad.zero_()

    def to(self, model):
        """In-place load of the Leap state state to model state_dict.

        Arguments:
            model (torch.nn.Module): target model to initialize with meta-learned initialization.
        """
        model.load_state_dict(clone_state_dict(self.state))

    def init_task(self, increment=True):
        """Initialize running variables for task training.

        Arguments:
            increment (bool): increment meta-batch task counter for call to normalize (default=True).
        """
        self.updater.initialize()
        if increment:
            self._task_counter += 1

    def normalize(self, reset_count=True):
        """Divide meta gradient by number of tasks encountered since last call to normalize.

        Arguments:
            reset_count (bool): reset meta-batch task counter.
        """
        assert self._task_counter != 0, "task counter is 0: call 'init_task' during meta training"

        for param in self.parameters():
            param.grad.data.div_(self._task_counter)

        if reset_count:
            self._task_counter = 0

    def update(self, loss, model, hook=None):
        """Increment the Leap meta-gradient.

        Arguments:
            loss (torch.Tensor): a 1-d loss tensor.
            model (torch.nn.Module): the model whose current parameters are to be used to increment the meta gradient.
            hook (func): an hook applied to the calculated gradient before incrementing the meta gradient (optional).
        """
        curr_state = clone_state_dict(model.state_dict(keep_vars=True))
        curr_loss = loss.clone()
        self.updater(curr_loss, curr_state, hook=hook)

    def named_parameters(self):
        """Iterator over named parameters in the Leap state dictionary."""
        for name, param in self.state.items():
            if param.requires_grad:
                yield name, param

    def parameters(self):
        """Iterator over parameters in state"""
        for _, param in self.named_parameters():
            yield param
