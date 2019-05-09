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

"""Helpers for manipulating stored data"""
# pylint: disable=invalid-name,too-many-locals,too-many-instance-attributes, redefined-builtin
import os
from os.path import join
import glob
import numpy as np


def load_args(log_path):
    """Returns the args.log string from an experiment root path."""
    args_file = join(log_path, 'args.log')
    with open(args_file, 'r') as af:
        args = af.read()
    return args


def load_results(file):
    """Load results from a log file into a dictionary."""
    all_accs = []
    all_losses = []
    mean_loss = []
    mean_meta_loss = []
    mean_acc = []
    steps = []

    with open(file, 'r') as fo:
        for line in fo:
            line = line.split(',')
            step, meta_loss, loss, acc, losses, accs = line
            step = int(step)
            loss = float(loss)
            acc = float(acc)
            losses = losses.split(';')
            losses = [float(l) for l in losses[:-1]]
            accs = accs.split(';')
            accs = [float(a) for a in accs[:-1]]

            if meta_loss is not None:
                meta_loss = float(meta_loss)
                mean_meta_loss.append(meta_loss)
            steps.append(step)
            mean_loss.append(loss)
            mean_acc.append(acc)
            all_losses.append(losses)
            all_accs.append(accs)

    out = {'step': steps,
           'loss': mean_loss,
           'acc': mean_acc,
           'loss_curve': all_losses,
           'acc_curve': all_accs}

    if mean_meta_loss:
        out['meta_loss'] = mean_meta_loss

    return out


#################################################################################
class FileHandler:

    """Loads logged data into format for plotting and analysis.

    Arguments:
        log_path (str): path to experiment root.
        name (str, None): name to use as label in plots.
    """

    def __init__(self, log_path, name=None):
        if name is None:
            name = os.path.basename(log_path)
        self.name = name
        self.log_path = log_path
        self.log_files = glob.glob(join(log_path, '*'))

        self.args = load_args(log_path)
        self.res_files = sorted(filter(lambda x: 'results_' in x, self.log_files))
        self.train_files = sorted(filter(lambda x: 'train_' in x, self.res_files))
        self.val_files = sorted(filter(lambda x: 'val_' in x, self.res_files))
        self.test_files = sorted(filter(lambda x: 'test_' in x, self.res_files))

        self.train_train_data = None
        self.train_eval_data = None
        self.val_train_data = None
        self.val_eval_data = None
        self.test_train_data = None
        self.test_eval_data = None

        if self.train_files:
            self.train_train_data = load_results(self.train_files[0])
            self.train_eval_data = load_results(self.train_files[1])

        if self.val_files:
            self.val_train_data, self.val_eval_data = self.load_eval(self.val_files)

        if self.test_files:
            self.test_train_data, self.test_eval_data = self.load_eval(self.test_files)

    @staticmethod
    def load_eval(files):
        """Load data from a evaluation tasks

        Arguments:
            files (list): list of files from val or test set to load.
        """
        train_data = []
        eval_data = []
        files = files.copy()
        assert len(files) % 2 == 0, 'files not in pairs of (train, eval). Cannot parse.'
        while files:
            train_file = files.pop(0)
            eval_file = files.pop(0)
            train_data.append(load_results(train_file))
            eval_data.append(load_results(eval_file))
        return train_data, eval_data

    def agg_val(self, metric, type):
        """Returns a numpy array with data for each validation task

        Arguments:
            metric (str): 'loss' or 'acc'.
            type (str): 'train' or 'eval'.
        """
        return self._agg('val', metric, type)

    def agg_test(self, metric, type):
        """Returns a numpy array with metric data for each test task
        Arguments:
            metric (str): 'loss' or 'acc'.
            type (str): 'train' or 'eval'.
        """
        return self._agg('test', metric, type)

    def plot_train(self, metric='acc', type='val', ax=None, **kwargs):
        """Utility for plotting train task performance over meta-training.
        Plots either loss or accuracy on training or
        test set for meta-training tasks.

        Arguments:
            metric (str): 'loss' or 'acc'.
            type (str): 'train' or 'eval'.
            ax (matplotlib.pyplot.ax): ax object (default=None)
            **kwargs (kwargs): optional arguments to plot.
        """
        label = kwargs.pop('label', self.name)
        x = self.train_train_data['step']
        y = getattr(self, 'train_{}_data'.format(type))[metric]
        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(1)
        ax.plot(x, y, label=label, **kwargs)
        return ax

    def plot_val(self, metric='acc', type='val', std=1., ax=None, **kwargs):
        """Utility for plotting val task performance over meta-training.
        Plots either loss or accuracy on training or
        test set for meta-training tasks.

        Arguments:
            metric (str): 'loss' or 'acc'.
            type (str): 'train' or 'eval'.
            std (float, None): number of standard deviations to use for shading.
            ax (matplotlib.pyplot.ax): ax object (default=None)
            **kwargs (kwargs): optional arguments to plot. If using std,
                'color' is shared between plot and fill_between.
                'alpha' is sent to fill_between.
                'linewidth' is sent to plot; set to 0 in fill_between.
        """
        return self._plot_holdout('val', metric, type, std, ax, **kwargs)

    def plot_test(self, metric='acc', type='val', std=1., ax=None, **kwargs):
        """Utility for plotting val task performance over meta-training.
        Plots either loss or accuracy on training or
        test set for meta-training tasks.

        Arguments:
            metric (str): 'loss' or 'acc'.
            type (str): 'train' or 'eval'.
            std (float, None): number of standard deviations to use for shading.
            ax (matplotlib.pyplot.ax): ax object (default=None)
            **kwargs (kwargs): optional arguments to plot. If using std,
                'color' is shared between plot and fill_between
                'alpha' is sent to fill_between.
                'linewidth' is sent to plot; set to 0 in fill_between.
        """
        return self._plot_holdout('test', metric, type, std, ax, **kwargs)

    def _agg(self, holdout, metric, type):
        y = getattr(self, '{}_{}_data'.format(holdout, type))
        y = [y_task[metric] for y_task in y]
        return np.stack(y, axis=1)

    def _plot_holdout(self, holdout, metric='acc', type='val', std=1., ax=None, **kwargs):
        label = kwargs.pop('label', self.name)
        x = getattr(self, '{}_{}_data'.format(holdout, type))[0]['step']

        y = self._agg(holdout, metric, type)
        y_mean = y.mean(axis=1)
        y_std = y.std(axis=1)

        if ax is None:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(1)

        alpha = kwargs.pop('alpha', 0.1)
        color = kwargs.pop('color', None)
        ax.plot(x, y_mean, label=label, color=color, **kwargs)
        if std:
            ax.fill_between(x, y_mean - std * y_std, y_mean + std * y_std,
                            linewidth=0., alpha=alpha, color=color)
        return ax
