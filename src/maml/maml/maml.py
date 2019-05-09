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

"""MAML

APIs for the MAML meta-learner.

"""
import torch.nn as nn
from .utils import build_dict, load_state_dict, build_iterator, Res, AggRes


def maml_inner_step(input, output, model, optimizer, criterion, create_graph):
    """Create a computation graph through the gradient operation

    Arguments:
        input (torch.Tensor): input tensor.
        output (torch.Tensor): target tensor.
        model (torch.nn.Module): task learner.
        optimizer (maml.optim): optimizer for inner loop.
        criterion (func): loss criterion.
        create_graph (bool): create graph through gradient step.
    """
    new_parameters = None

    prediction = model(input)
    loss = criterion(prediction, output)
    loss.backward(create_graph=create_graph, retain_graph=create_graph)

    if create_graph:
        _, new_parameters = optimizer.step(retain_graph=create_graph)
    else:
        optimizer.step(retain_graph=create_graph)

    return loss, prediction, new_parameters


def maml_task(data_inner, data_outer, model, optimizer, criterion, create_graph):
    """Adapt model parameters to task and use adapted params to predict new samples

    Arguments:
        data_inner (iterable): list of input-output for task adaptation.
        data_outer (iterable): list of input-output for task validation.
        model (torch.nn.Module): task learner.
        optimizer (maml.optim): optimizer for inner loop.
        criterion (func): loss criterion.
        create_graph (bool): create graph through gradient step.
    """
    original_parameters = model.state_dict(keep_vars=True)
    device = next(model.parameters()).device

    # Adaptation of parameters to task
    train_res = Res()
    for i, (input, output) in enumerate(data_inner):
        input = input.to(device, non_blocking=True)
        output = output.to(device, non_blocking=True)

        loss, prediction, new_params = maml_inner_step(input, output, model, optimizer, criterion, create_graph)

        train_res.log(loss.item(), prediction, output)

        if create_graph:
            load_state_dict(model, build_dict([n for n, _ in model.named_parameters()], new_params))

        for p in original_parameters.values():
            p.grad = None

    # Run with adapted parameters on task
    val_res = Res()
    predictions = []
    for i, (input, output) in enumerate(data_outer):
        input = input.to(device, non_blocking=True)
        output = output.to(device, non_blocking=True)

        prediction = model(input)
        predictions.append(prediction)

        batch_loss = criterion(prediction, output)
        loss += batch_loss

        val_res.log(batch_loss.item(), prediction, output)

    loss = loss / (i + 1)
    load_state_dict(model, original_parameters)

    return loss, predictions, (train_res, val_res)


def maml_outer_step(task_iterator, model, optimizer_cls, criterion, return_predictions=True,
                    return_results=True, create_graph=True, **optimizer_kwargs):
    """MAML objective.

    Run MAML on a batch of tasks.


    Arguments:
        task_iterator (iterator): data sampler for K tasks. Of the format
            [task1, task2, task3] where each task is of the format
            task1 = (data_iterator_inner, data_iterator_outer) and each
            data_iterator_ = [(input_batch1, target_batch1), ...]

            ::note::
                the inner data_iterator defines the number of gradient

        model (Module): task learner.
        optimizer_cls (maml.optim.SGD, maml.optim.Adam): inner optimizer class.
            Must allow backpropagation through gradient step.
        criterion (func): loss criterion.
        return_predictions (bool): whether to return.
        return_results (bool): return accumulated meta-data.
        create_graph (bool): create computational graph through gradient step.
        optimizer_kwargs (kwargs): kwargs to optimizer.
    """
    loss = 0
    predictions, results = [], []
    for i, task in enumerate(task_iterator):
        inner_iterator, outer_iterator = task
        task_optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

        task_loss, task_predictions, task_res = maml_task(
            inner_iterator, outer_iterator, model, task_optimizer, criterion, create_graph)

        loss += task_loss

        predictions.append(task_predictions)
        results.append(task_res)

    loss = loss / (i + 1)
    results = AggRes(results)

    out = [loss]
    if return_predictions:
        out.append(predictions)
    if return_results:
        out.append(results)
    return out

###############################################################################


class MAML(nn.Module):

    """MAML

    Class Instance for the MAML objective

    Arguments:
        model (torch.nn.Module): task learner.
        optimizer_cls (maml.optim): task optimizer. Note: must allow backpropagation through gradient steps.
        criterion (func): loss criterion.
        tensor (bool): whether meta mini-batches come as a tensor or as a list of dataloaders.
        inner_bsz (int): if tensor=True, batch size in inner loop.
        outer_bsz (int): if tensor=True, batch size in outer loop.
        inner_steps (int): if tensor=True, number of steps in inner loop.
        outer_steps (int): if tensor=True, number of steps in outer loop.

    Example:
        >>> loss = maml.forward(task_iterator)
        >>> loss.backward()
        >>> meta_optimizer.step()
        >>> meta_optimizer.zero_grad()
    """

    def __init__(self, model, optimizer_cls, criterion, tensor,
                 inner_bsz=None, outer_bsz=None, inner_steps=None,
                 outer_steps=None, **optimizer_kwargs):
        super(MAML, self).__init__()

        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.criterion = criterion

        self.tensor = tensor
        self.inner_bsz = inner_bsz
        self.outer_bsz = outer_bsz
        self.inner_steps = inner_steps
        self.outer_steps = outer_steps

        if tensor:
            assert inner_bsz is not None, 'set inner_bsz with tensor=True'
            assert outer_bsz is not None, 'set outer_bsz with tensor=True'
            assert inner_steps is not None, 'set inner_steps with tensor=True'
            assert outer_steps is not None, 'set outer_steps with tensor=True'

    def forward(self, inputs, return_predictions=True, return_results=True, create_graph=True):
        task_iterator = inputs if not self.tensor else [
            build_iterator(i, self.inner_bsz, self.outer_bsz, self.inner_steps, self.outer_steps)
            for i in inputs]
        return maml_outer_step(
            task_iterator=task_iterator,
            model=self.model,
            optimizer_cls=self.optimizer_cls,
            criterion=self.criterion,
            return_predictions=return_predictions,
            return_results=return_results,
            create_graph=create_graph,
            **self.optimizer_kwargs)
