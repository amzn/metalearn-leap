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

"""Meta-learners for Omniglot experiment."""
import random
from abc import abstractmethod
from torch import nn
from torch import optim

from leap import Leap
from leap.utils import clone_state_dict
import maml

from utils import Res, AggRes


class BaseWrapper(object):

    """Generic training wrapper.

    Arguments:
        criterion (func): loss criterion to use.
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
    """

    def __init__(self, criterion, model, optimizer_cls, optimizer_kwargs):
        self.criterion = criterion
        self.model = model

        self.optimizer_cls = optim.SGD if optimizer_cls.lower() == 'sgd' else optim.Adam
        self.optimizer_kwargs = optimizer_kwargs

    def __call__(self, tasks, meta_train=True):
        return self.run_tasks(tasks, meta_train=meta_train)

    @abstractmethod
    def _partial_meta_update(self, loss, final):
        """Meta-model specific meta update rule.

        Arguments:
            loss (nn.Tensor): loss value for given mini-batch.
            final (bool): whether iteration is the final training step.
        """
        NotImplementedError('Implement in meta-learner class wrapper.')

    @abstractmethod
    def _final_meta_update(self):
        """Meta-model specific meta update rule."""
        NotImplementedError('Implement in meta-learner class wrapper.')

    def run_tasks(self, tasks, meta_train):
        """Train on a mini-batch tasks and evaluate test performance.

        Arguments:
            tasks (list, torch.utils.data.DataLoader): list of task-specific dataloaders.
            meta_train (bool): whether current run in during meta-training.
        """
        results = []
        for task in tasks:
            task.dataset.train()
            trainres = self.run_task(task, train=True, meta_train=meta_train)
            task.dataset.eval()
            valres = self.run_task(task, train=False, meta_train=False)
            results.append((trainres, valres))
        ##
        results = AggRes(results)

        # Meta gradient step
        if meta_train:
            self._final_meta_update()

        return results

    def run_task(self, task, train, meta_train):
        """Run model on a given task.

        Arguments:
            task (torch.utils.data.DataLoader): task-specific dataloaders.
            train (bool): whether to train on task.
            meta_train (bool): whether to meta-train on task.
        """
        optimizer = None
        if train:
            self.model.train()
            optimizer = self.optimizer_cls(
                self.model.parameters(), **self.optimizer_kwargs)
        else:
            self.model.eval()

        return self.run_batches(task, optimizer, train=train, meta_train=meta_train)

    def run_batches(self, batches, optimizer, train=False, meta_train=False):
        """Iterate over task-specific batches.

        Arguments:
            batches (torch.utils.data.DataLoader): task-specific dataloaders.
            optimizer (torch.nn.optim): optimizer instance if training is True.
            train (bool): whether to train on task.
            meta_train (bool): whether to meta-train on task.
        """
        device = next(self.model.parameters()).device

        res = Res()
        N = len(batches)
        for n, (input, target) in enumerate(batches):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # Evaluate model
            prediction = self.model(input)
            loss = self.criterion(prediction, target)

            res.log(loss=loss.item(), pred=prediction, target=target)

            # TRAINING #
            if not train:
                continue

            final = (n+1) == N
            loss.backward()

            if meta_train:
                self._partial_meta_update(loss, final)

            optimizer.step()
            optimizer.zero_grad()

            if final:
                break
        ###
        res.aggregate()
        return res


class LeapWrapper(BaseWrapper):

    """Wrapper around the Leap meta-learner.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
        meta_kwargs (dict): kwargs to pass to meta-learner upon construction.
        criterion (func): loss criterion to use.
    """

    def __init__(self, model, optimizer_cls, meta_optimizer_cls, optimizer_kwargs,
                 meta_optimizer_kwargs, meta_kwargs, criterion):
        super(LeapWrapper, self).__init__(criterion, model, optimizer_cls, optimizer_kwargs)
        self.meta = Leap(model, **meta_kwargs)

        self.meta_optimizer_cls = optim.SGD if meta_optimizer_cls.lower() == 'sgd' else optim.Adam
        self.meta_optimizer = self.meta_optimizer_cls(self.meta.parameters(), **meta_optimizer_kwargs)

    def _partial_meta_update(self, l, final):
        self.meta.update(l, self.model)

    def _final_meta_update(self):
        self.meta.normalize()
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()

    def run_task(self, task, train, meta_train):
        if meta_train:
            self.meta.init_task()

        if train:
            self.meta.to(self.model)

        return super(LeapWrapper, self).run_task(task, train, meta_train)


class MAMLWrapper(object):

    """Wrapper around the MAML meta-learner.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
        criterion (func): loss criterion to use.
    """

    def __init__(self, model, optimizer_cls, meta_optimizer_cls, optimizer_kwargs,
                 meta_optimizer_kwargs, criterion):
        self.criterion = criterion
        self.model = model

        self.optimizer_cls = maml.SGD if optimizer_cls.lower() == 'sgd' else maml.Adam

        self.meta = maml.MAML(optimizer_cls=self.optimizer_cls, criterion=criterion,
                              model=model, tensor=False, **optimizer_kwargs)

        self.meta_optimizer_cls = optim.SGD if meta_optimizer_cls.lower() == 'sgd' else optim.Adam

        self.optimizer_kwargs = optimizer_kwargs
        self.meta_optimizer = self.meta_optimizer_cls(self.meta.parameters(), **meta_optimizer_kwargs)

    def __call__(self, meta_batch, meta_train):
        tasks = []
        for t in meta_batch:
            t.dataset.train()
            inner = [b for b in t]
            t.dataset.eval()
            outer = [b for b in t]
            tasks.append((inner, outer))
        return self.run_meta_batch(tasks, meta_train=meta_train)

    def run_meta_batch(self, meta_batch, meta_train):
        """Run on meta-batch.

        Arguments:
            meta_batch (list): list of task-specific dataloaders
            meta_train (bool): meta-train on batch.
        """
        loss, results = self.meta(meta_batch, return_predictions=False, return_results=True, create_graph=meta_train)
        if meta_train:
            loss.backward()
            self.meta_optimizer.step()
            self.meta_optimizer.zero_grad()

        return results


class NoWrapper(BaseWrapper):

    """Wrapper for baseline without any meta-learning.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        criterion (func): loss criterion to use.
    """
    def __init__(self, model, optimizer_cls, optimizer_kwargs, criterion):
        super(NoWrapper, self).__init__(criterion, model, optimizer_cls, optimizer_kwargs)
        self._original = clone_state_dict(model.state_dict(keep_vars=True))

    def __call__(self, tasks, meta_train=False):
        return super(NoWrapper, self).__call__(tasks, meta_train=False)

    def run_task(self, *args, **kwargs):
        out = super(NoWrapper, self).run_task(*args, **kwargs)
        self.model.load_state_dict(self._original)
        return out

    def _partial_meta_update(self, loss, final):
        pass

    def _final_meta_update(self):
        pass


class _FOWrapper(BaseWrapper):

    """Base wrapper for First-order MAML and Reptile.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
        criterion (func): loss criterion to use.
    """

    _all_grads = None

    def __init__(self, model, optimizer_cls, meta_optimizer_cls, optimizer_kwargs,
                 meta_optimizer_kwargs, criterion):
        super(_FOWrapper, self).__init__(criterion, model, optimizer_cls, optimizer_kwargs)
        self.meta_optimizer_cls = optim.SGD if meta_optimizer_cls.lower() == 'sgd' else optim.Adam
        self.meta_optimizer_kwargs = meta_optimizer_kwargs

        self._counter = 0
        self._updates = None
        self._original = clone_state_dict(self.model.state_dict(keep_vars=True))

        params = [p for p in self._original.values() if getattr(p, 'requires_grad', False)]
        self.meta_optimizer = self.meta_optimizer_cls(params, **meta_optimizer_kwargs)

    def run_task(self, task, train, meta_train):
        if meta_train:
            self._counter += 1
        if train:
            self.model.load_state_dict(self._original)
        return super(_FOWrapper, self).run_task(task, train, meta_train)

    def _partial_meta_update(self, loss, final):
        if not final:
            return

        if self._updates is None:
            self._updates = {}
            for n, p in self._original.items():
                if not getattr(p, 'requires_grad', False):
                    continue

                if p.size():
                    self._updates[n] = p.new(*p.size()).zero_()
                else:
                    self._updates[n] = p.clone().zero_()

        for n, p in self.model.state_dict(keep_vars=True).items():
            if n not in self._updates:
                continue

            if self._all_grads is True:
                self._updates[n].add_(p.data)
            else:
                self._updates[n].add_(p.grad.data)

    def _final_meta_update(self):
        for n, p in self._updates.items():
            p.data.div_(self._counter)

        for n, p in self._original.items():
            if n not in self._updates:
                continue

            if self._all_grads:
                p.grad = p.data - self._updates[n].data
            else:
                p.grad = self._updates[n]

        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
        self._counter = 0
        self._updates = None


class ReptileWrapper(_FOWrapper):

    """Wrapper for Reptile.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
        criterion (func): loss criterion to use.
    """

    _all_grads = True

    def __init__(self, *args, **kwargs):
        super(ReptileWrapper, self).__init__(*args, **kwargs)


class FOMAMLWrapper(_FOWrapper):
    """Wrapper for FOMAML.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
        criterion (func): loss criterion to use.
    """

    _all_grads = False

    def __init__(self, *args, **kwargs):
        super(FOMAMLWrapper, self).__init__(*args, **kwargs)


class FtWrapper(BaseWrapper):

    """Wrapper for Multi-headed finetuning.

    This wrapper differs from others in that it blends batches from all tasks into a single epoch.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        criterion (func): loss criterion to use.
    """

    def __init__(self, model, optimizer_cls, optimizer_kwargs, criterion):
        super(FtWrapper, self).__init__(criterion, model, optimizer_cls, optimizer_kwargs)
        # We use the same inner optimizer throughout
        self.optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)

    @staticmethod
    def gen_multitask_batches(tasks, train):
        """Generates one batch iterator across all tasks. Use mode to switch between train and eval."""
        iterator_id = 0
        all_batches = []
        for task_id, iterator in tasks:
            if train:
                iterator.dataset.train()
            else:
                iterator.dataset.eval()

            for batch in iterator:
                all_batches.append((iterator_id, task_id, batch))
            iterator_id += 1

        if train:
            random.shuffle(all_batches)

        return all_batches

    def run_tasks(self, tasks, meta_train):
        original = None
        if not meta_train:
            original = clone_state_dict(self.model.state_dict(keep_vars=True))

            # Non-transductive task evaluation for fair comparison
            for module in self.model.modules():
                if hasattr(module, 'reset_running_stats'):
                    module.reset_running_stats()

        # Training #
        all_batches = self.gen_multitask_batches(tasks, train=True)
        trainres = self.run_multitask(all_batches, train=True)

        # Eval #
        all_batches = self.gen_multitask_batches(tasks, train=False)
        valres = self.run_multitask(all_batches, train=False)

        results = AggRes(zip(trainres, valres))

        if not meta_train:
            self.model.load_state_dict(original)

        return results

    def _partial_meta_update(self, l, final):
        return

    def _final_meta_update(self):
        return

    def run_multitask(self, batches, train):
        """Train on task in multi-task mode

        This is equivalent to the run_task method but differs in that
        batches are assumed to be mixed from different tasks.
        """
        N = len(batches)

        if train:
            self.model.train()
        else:
            self.model.eval()

        device = next(self.model.parameters()).device

        res = {}
        for n, (iterator_id, task_id, (input, target)) in enumerate(batches):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            prediction = self.model(input, task_id)

            loss = self.criterion(prediction, target)

            if iterator_id not in res:
                res[iterator_id] = Res()

            res[iterator_id].log(loss=loss.item(), pred=prediction, target=target)

            # TRAINING #
            if not train:
                continue

            final = (n + 1) == N
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if final:
                break
        ###
        res = [r[1] for r in sorted(res.items(), key=lambda r: r[0])]
        for r in res:
            r.aggregate()

        return res
