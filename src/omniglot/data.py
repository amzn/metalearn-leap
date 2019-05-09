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

"""Data loading facilities for Omniglot experiment."""
import random
import os
from os.path import join

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision.datasets.omniglot import Omniglot as _Omniglot
from torchvision.datasets.utils import list_dir, list_files
from torchvision import transforms


TRAIN = 30
VAL = 10
TEST = 10

HOLD_OUT = 5

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.RandomAffine((0, 360), (0.2, 0.2), (0.8, 1.2)),
    transforms.ToTensor(),
])


####################################################################################

class DataContainer(object):

    """Data container class for Omniglot

    Arguments:
        root (str): root of dataset.
        num_pretrain_alphabets (int): number of alphabets to use for meta-training.
        num_classes (int): number of classes to enforce per task (optional).
        transform (func): transformation to apply to each input sample.
        seed (int): seed used to shuffle alphabets when creating train/val/test splits.
        **kwargs (dict): keyword arguments to pass to the torch.utils.data.DataLoader
    """

    folder = "omniglot-py"
    target_folder = "images_resized"

    def __init__(self, root="./data", num_pretrain_alphabets=1, num_classes=None,
                 transform=DEFAULT_TRANSFORM, seed=1, **kwargs):
        self.root = root
        self.num_pretrain_alphabets = num_pretrain_alphabets
        self.transform = transform
        self.seed = seed
        self.kwargs = kwargs

        path = join(join(os.path.expanduser(self.root), self.folder), self.target_folder)
        alphabets = list_dir(path)
        if num_classes:
            alphabets = [a for a in alphabets if len(list_dir(join(path, a))) >= num_classes]
            assert self.num_pretrain_alphabets + TEST < len(alphabets), 'cannot create test set'

        random.seed(self.seed)
        random.shuffle(alphabets)

        trs = self.num_pretrain_alphabets
        tes = trs + TEST

        train = alphabets[:trs]
        test = alphabets[trs:tes]
        val = alphabets[tes:]

        trainset = [SubOmniglot(root, [t], num_classes, HOLD_OUT, transform=transform) for t in train]
        testset = [SubOmniglot(root, [v], num_classes, HOLD_OUT, transform=transform) for v in test]
        valset = [SubOmniglot(root, [v], num_classes, HOLD_OUT, transform=transform) for v in val]

        self.alphabets = alphabets
        self.alphabets_train = train
        self.alphabets_test = test
        self.alphabets_val = val

        self.data_train = trainset
        self.data_test = testset
        self.data_val = valset

    def get_loader(self, task, batch_size, iterations):
        """Returns a DataLoader for given configuration.

        Arguments:
            task (SubOmniglot): A SubOmniglot instance to pass to a DataLoader instance.
            batch_size (int): batch size in data loader.
            iterations (int): number of batches.
        """
        return DataLoader(task, batch_size, sampler=RandomSampler(task, iterations, batch_size), **self.kwargs)

    def train(self, meta_batch_size, batch_size, iterations, return_idx=False):
        """Generator meta-train batch

        Arguments:
            meta_batch_size (int): number of tasks in batch.
            batch_size (int): number of samples in each batch in the inner (task) loop.
            iterations (int): number of training steps on each task.
            return_idx (int): return task ids (default=False).
        """
        n_tasks = len(self.data_train)

        if n_tasks == 1:
            tasks = zip([0] * meta_batch_size, self.data_train * meta_batch_size)
        else:
            tasks = []
            task_ids = list(range(n_tasks))
            while True:
                random.shuffle(task_ids)
                tasks.extend([(i, self.data_train[i]) for i in task_ids])
                if len(tasks) >= meta_batch_size:
                    break
            tasks = tasks[:meta_batch_size]

        task_ids, task_data = zip(*tasks)
        task_data = [self.get_loader(t, batch_size, iterations)for t in task_data]

        if return_idx:
            return list(zip(task_ids, task_data))
        return task_data

    def val(self, batch_size, iterations, return_idx=False):
        """Generator meta-validation batch

        Arguments:
            batch_size (int): number of samples in each batch in the inner (task) loop.
            iterations (int): number of training steps on each task.
            return_idx (int): return task ids (default=False).
        """
        n = len(self.data_train)

        tsk = [i+n for i in range(len(self.data_val))]
        tasks = [self.get_loader(d, batch_size, iterations) for d in self.data_val]

        if return_idx:
            return list(zip(tsk, tasks))
        return tasks

    def test(self, batch_size, iterations, return_idx=False):
        """Generator meta-test batch

        Arguments:
            batch_size (int): number of samples in each batch in the inner (task) loop.
            iterations (int): number of training steps on each task.
            return_idx (int): return task ids (default=False).
        """
        n = len(self.data_train) + len(self.data_val)

        tsk = [i+n for i in range(len(self.data_test))]
        tasks = [self.get_loader(d, batch_size, iterations) for d in self.data_test]

        if return_idx:
            return list(zip(tsk, tasks))
        return tasks


class SubOmniglot(_Omniglot):

    """Data class for Omniglot that subsamples a specified number of alphabets.

    Arguments:
        root (str): root of the Omniglot dataset.
        alphabets (int): number of alphabets to use in the creation of the dataset.
        num_classes (int): number of classes to enforce per task (optional).
        hold_out (int): number of samples per character to hold for validation set (optional).
        transform (func): transformation to apply to each input sample.
        seed (int): seed used to shuffle alphabets when creating train/val/test splits.
    """

    folder = "omniglot-py"
    target_folder = "images_resized"

    def __init__(self, root, alphabets, num_classes=None, hold_out=None, transform=None, seed=None):
        self.root = join(os.path.expanduser(root), self.folder)
        self.alphabets = alphabets
        self.num_classes = num_classes
        self.hold_out = hold_out
        self.transform = transform
        self.target_transform = None
        self.seed = seed

        self.target_folder = join(self.root, self.target_folder)
        self._alphabets = [a for a in list_dir(self.target_folder) if a in self.alphabets]
        self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
                                for a in self._alphabets], [])

        if seed:
            random.seed(seed)

        random.shuffle(self._characters)

        if self.num_classes:
            self._characters = self._characters[:num_classes]

        self._character_images = [
            [(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
            for idx, character in enumerate(self._characters)
        ]

        self._train_character_images = []
        self._val_character_images = []
        for idx, character in enumerate(self._characters):
            train_characters = []
            val_characters = []
            for img_count, image in enumerate(list_files(join(self.target_folder, character), '.png')):
                if hold_out and img_count < hold_out:
                    val_characters.append((image, idx))
                else:
                    train_characters.append((image, idx))
            self._train_character_images.append(train_characters)
            self._val_character_images.append(val_characters)

        self._flat_train_character_images = sum(self._train_character_images, [])
        self._flat_val_character_images = sum(self._val_character_images, [])

        self._train = True
        self._set_images()

    def train(self):
        """Train mode"""
        self._train = True
        self._set_images()

    def eval(self):
        """Eval mode"""
        self._train = False
        self._set_images()

    def _set_images(self):
        """Set images"""
        if self.train:
            self._flat_character_images = self._flat_train_character_images
        else:
            self._flat_character_images = self._flat_val_character_images


class RandomSampler(Sampler):
    r"""Samples elements randomly with replacement (if iterations > data set).

    Arguments:
        data_source (Dataset): dataset to sample from
        iterations (int): number of samples to return on each call to __iter__
        batch_size (int): number of samples in each batch
    """

    def __init__(self, data_source, iterations, batch_size):
        self.data_source = data_source
        self.iterations = iterations
        self.batch_size = batch_size

    def __iter__(self):
        if self.data_source._train:
            idx = torch.randperm(self.iterations * self.batch_size) % len(self.data_source)
        else:
            idx = torch.randperm(len(self.data_source))
        return iter(idx.tolist())

    def __len__(self):  # pylint: disable=protected-access
        return self.iterations * self.batch_size if self.data_source._train else len(self.data_source)
