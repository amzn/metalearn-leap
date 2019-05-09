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

"""Classifier for Omniglot experiment."""
import torch.nn as nn
from wrapper import LeapWrapper, MAMLWrapper, NoWrapper, FtWrapper, FOMAMLWrapper, ReptileWrapper


def get_model(args, criterion):
    """Construct model from main args"""
    model = OmniConv(
        args.classes,
        args.num_layers,
        args.kernel_size,
        args.num_filters,
        args.imsize,
        args.padding,
        args.max_pool,
        args.batch_norm,
        args.multi_head,
    )

    if args.cuda:
        model = model.cuda()

    if args.log_ival > 0:
        print(model)

    if args.meta_model.lower() == 'leap':
        return LeapWrapper(
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            args.meta_kwargs,
            criterion,
        )

    if args.meta_model.lower() == 'no':
        return NoWrapper(
            model,
            args.inner_opt,
            args.inner_kwargs,
            criterion,
        )

    if args.meta_model.lower() == 'ft':
        return FtWrapper(
            model,
            args.inner_opt,
            args.inner_kwargs,
            criterion,
        )

    if args.meta_model.lower() == 'fomaml':
        return FOMAMLWrapper(
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            criterion,
        )

    if args.meta_model.lower() == 'reptile':
        return ReptileWrapper(
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            criterion,
        )

    if args.meta_model.lower() == 'maml':
        return MAMLWrapper(
            model,
            args.inner_opt,
            args.outer_opt,
            args.inner_kwargs,
            args.outer_kwargs,
            criterion,
        )
    raise NotImplementedError('Meta-learner {} unknown.'.format(args.meta_model.lower()))


################################################################################


class UnSqueeze(nn.Module):

    """Create channel dim if necessary."""

    def __init__(self):
        super(UnSqueeze, self).__init__()

    def forward(self, input):
        """Creates channel dimension on a 3-d tensor. Null-op if input is a 4-d tensor.

        Arguments:
            input (torch.Tensor): tensor to unsqueeze.
        """
        if input.dim() == 4:
            return input
        return input.unsqueeze(1)


class Squeeze(nn.Module):

    """Undo excess dimensions"""

    def __init__(self):  # pylint: disable=useless-super-delegation
        super(Squeeze, self).__init__()

    def forward(self, input):
        """Squeeze singular dimensions of an input tensor.

        Arguments:
            input (torch.Tensor): tensor to unsqueeze.
        """
        if input.size(0) != 0:
            return input.squeeze()
        input = input.squeeze()
        return input.view(1, *input.size())


class OmniConv(nn.Module):

    """ConvNet classifier.

    Arguments:
        nclasses (int): number of classes to predict in each alphabet
        nlayers (int): number of convolutional layers (default=4).
        kernel_size (int): kernel size in each convolution (default=3).
        num_filters (int): number of output filters in each convolution (default=64)
        imsize (tuple): tuple of image height and width dimension.
        padding (bool, int, tuple): padding argument to convolution layers (default=True).
        max_pool(bool): use max pooling in each convolution layer (default=True)
        batch_norm (bool): use batch normalization in each convolution layer (default=True).
        multi_head (bool): multi-headed training (default=False).
    """

    def __init__(self, nclasses, nlayers=4, kernel_size=3,
                 num_filters=64, imsize=(28, 28), padding=True,
                 max_pool=True, batch_norm=True, multi_head=False):
        super(OmniConv, self).__init__()
        self.nlayers = nlayers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.imsize = imsize
        self.max_pool = max_pool
        self.batch_norm = batch_norm
        self.multi_head = multi_head

        def conv_block(nin):
            block = [nn.Conv2d(nin, num_filters, kernel_size, padding=padding)]
            if max_pool:
                block.append(nn.MaxPool2d(2))
            if batch_norm:
                block.append(nn.BatchNorm2d(num_filters))
            block.append(nn.ReLU())
            return block

        layers = [UnSqueeze()]
        for i in range(nlayers):
            layers.extend(conv_block(1 if i == 0 else num_filters))

        if not max_pool:
            fsz = imsize[0] - 2 * nlayers if padding else imsize[0]
            layers.append(nn.AvgPool2d(fsz))

        layers.append(Squeeze())
        if not self.multi_head:
            layers.append(nn.Linear(num_filters, nclasses))
            self.model = nn.Sequential(*layers)
        else:
            self.conv = nn.Sequential(*layers)
            self.heads = nn.ModuleList([nn.Linear(num_filters, nclasses) for _ in range(50)])

    def forward(self, input, idx=None):
        if not self.multi_head:
            return self.model(input)
        input = self.conv(input)
        return self.heads[idx](input)
