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

"""Omniglot experiment accompanying https://arxiv.org/abs/1812.01054.

Executable script for training a meta-learner on Omniglot.
"""
import argparse
import time
import os
from os.path import join

from data import DataContainer

import torch
from torch import nn

from model import get_model
from utils import build_kwargs, log_status, write_train_res, write_val_res, unlink, consolidate

parser = argparse.ArgumentParser("Omniglot meta-learning script from https://arxiv.org/abs/1812.01054")

parser.add_argument('--root', type=str, default='./data', help="The number of classes to predict at any given draw")
parser.add_argument('--seed', type=int, default=8879, help="The seed to use")
parser.add_argument('--workers', type=int, default=0, help="Data-loading parallelism")

parser.add_argument('--num_pretrain', type=int, default=20, help="Number of tasks to meta-train on")
parser.add_argument('--classes', type=int, default=20, help="Number of classes in a task")

parser.add_argument('--meta_batch_size', type=int, default=20, help="Tasks per meta-batch")
parser.add_argument('--task_batch_size', type=int, default=20, help="Samples per task-batch")
parser.add_argument('--meta_train_steps', type=int, default=1000, help="Number of steps in the outer (meta) loop")
parser.add_argument('--task_train_steps', type=int, default=100, help="Number of steps in the inner (task) loop")
parser.add_argument('--task_val_steps', type=int, default=100, help="Number of steps when training on validation tasks")

parser.add_argument('--log_ival', type=int, default=1, help="Interval between logging to stdout")
parser.add_argument('--write_ival', type=int, default=1, help="Interval between logging to file")
parser.add_argument('--test_ival', type=int, default=20, help="Interval between evaluating on validation set")

parser.add_argument('--no_cuda', action='store_true', help="Don't use GPU acceleration")
parser.add_argument('--device', type=int, default=0, help="Index for GPU device")
parser.add_argument('--log_dir', type=str, default='./logs', help="Directory to write logs to")
parser.add_argument('--suffix', type=str, default='tmp', help="Name of experiment")
parser.add_argument('--overwrite', action='store_true', help='Allow overwrite of existing log dir (same suffix)')
parser.add_argument('--evaluate', action='store_true', help='Evaluate saved model')

parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size in conv layers')
parser.add_argument('--padding', type=int, default=1, help='Padding in conv layers')
parser.add_argument('--num_layers', type=int, default=4, help='Number of convolution layers in classifier')
parser.add_argument('--num_filters', type=int, default=64, help='Number of filters in each conv layer')
parser.add_argument('--no_max_pool', action='store_true', help='Turn off max pooling (switch to average pooling)')
parser.add_argument('--no_batch_norm', action='store_true', help='Turn off batch normalization')

parser.add_argument('--meta_model', type=str, default='leap', help='Meta-learner [leap, reptile, maml, fomaml, ft, no]')
parser.add_argument('--inner_opt', type=str, default='sgd', help='Optimizer in inner (task) loop: SGD or Adam')
parser.add_argument('--outer_opt', type=str, default='sgd', help='Optimizer in outer (meta) loop: SGD or Adam')
parser.add_argument('--inner_kwargs', nargs='+', default=['lr', '0.1'], help='Kwargs for inner optimizer')
parser.add_argument('--outer_kwargs', nargs='+', default=['lr', '0.1'], help='Kwargs for outer optimizer')
parser.add_argument('--meta_kwargs', nargs='+', default=[], help='Kwargs for meta learner')
args = parser.parse_args()

args.imsize = (28, 28)
args.cuda = not args.no_cuda
args.max_pool = not args.no_max_pool
args.batch_norm = not args.no_batch_norm

args.inner_kwargs = build_kwargs(args.inner_kwargs)
args.outer_kwargs = build_kwargs(args.outer_kwargs)
args.meta_kwargs = build_kwargs(args.meta_kwargs)
args.multi_head = args.meta_model.lower() == 'ft'


def pp(*inputs, **kwargs):
    """Print only if verbose is on"""
    if args.log_ival > 0:
        print(*inputs, **kwargs)


if args.cuda and not torch.cuda.is_available:
    raise ValueError("Cuda is not available. Run with --no_cuda")

if args.no_cuda and torch.cuda.is_available:
    pp("WARNING: Cuda is available, but running on CPU")

pp(args)

torch.manual_seed(args.seed)


def main():
    """Run script"""

    log_dir = os.path.join(args.log_dir, args.meta_model, args.suffix)

    data = DataContainer(
        root=args.root,
        num_pretrain_alphabets=args.num_pretrain,
        num_classes=args.classes,
        seed=args.seed,
        num_workers=args.workers,
        pin_memory=True,
    )

    ###############################################################################

    def evaluate(model, case, step):
        """Run final evaluation"""
        if args.write_ival > 0:
            torch.save(model, join(log_dir, 'model.pth.tar'))

        if case == 'test':
            iterator = data.test(args.task_batch_size, args.task_val_steps, args.multi_head)
        else:
            iterator = data.val(args.task_batch_size, args.task_val_steps, args.multi_head)

        pp('Evaluating on {} tasks'.format(case))

        results = []
        for i, task in enumerate(iterator):

            if args.write_ival > 0:
                task_model = torch.load(join(log_dir, 'model.pth.tar'))
            else:
                task_model = model

            t = time.time()
            task_results = task_model([task], meta_train=False)
            t = time.time() - t

            results.append(task_results)

            if args.log_ival > 0:
                log_status(task_results, 'task={}'.format(i), t)

        if args.log_ival > 0:
            log_status(consolidate(results), 'task avg', t)

        if args.write_ival > 0:
            write_val_res(results, step, case, log_dir)

        pp('Done')

    ###############################################################################

    criterion = nn.CrossEntropyLoss()

    if args.evaluate:
        model = torch.load(join(log_dir, 'model.pth.tar'))
        evaluate(model, 'test', 0)
        return

    model = get_model(args, criterion)

    ###############################################################################
    if args.write_ival > 0:
        if os.path.exists(log_dir):
            assert args.overwrite, "Path exists ({}). Use --overwrite or change suffix".format(log_dir)
            unlink(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        with open(join(log_dir, 'args.log'), 'w') as f:
            f.write("%r" % args)

        with open(join(log_dir, 'args.log'), 'w') as f:
            f.write("%r" % args)

    pp('Initiating meta-training')
    train_step = 0
    try:
        evaluate(model, 'val', train_step)

        t = time.time()
        while True:
            task_batch = data.train(args.meta_batch_size, args.task_batch_size, args.task_train_steps, args.multi_head)

            results = model(task_batch, meta_train=True)

            train_step += 1
            if train_step % args.write_ival == 0:
                write_train_res(results, train_step, log_dir)

            if train_step % args.test_ival == 0:
                evaluate(model, 'val', train_step)
                pp("Resuming training")

            if args.log_ival > 0 and train_step % args.log_ival == 0:
                t = (time.time() - t) / args.log_ival
                log_status(results, 'step={}'.format(train_step), t)
                t = time.time()

            if results.train_loss != results.train_loss:
                break

            if train_step == args.meta_train_steps:
                break

    except KeyboardInterrupt:
        pp('Meta-training stopped.')
    else:
        pp('Meta-training complete.')

    try:
        model = torch.load(join(log_dir, 'model.pth.tar'))
    except OSError:
        pp("No saved model. Using latest for final evaluation")

    evaluate(model, 'test', train_step)


if __name__ == '__main__':
    if args.cuda:
        with torch.cuda.device(args.device):
            main()
    else:
        main()
