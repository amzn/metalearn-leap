## Transferring Knowledge across Learning Processes  

[[Blog post]](https://medium.com/@flnr/transferring-knowledge-across-learning-processes-f6f63e9e6f46)  [[Paper]](https://arxiv.org/abs/1812.01054)

Original [PyTorch](https://pytorch.org/) implementation of the Leap meta-learner (https://arxiv.org/abs/1812.01054)
along with code for running the Omniglot experiment presented in the paper.

## License

This library is licensed under the Apache 2.0 License.

## Authors

Sebastian Flennerhag

## Install

This repository was developed against PyTorch v0.4 on Ubuntu 16.04 using Python 3.6. To install
Leap, clone the repo and install the source code:

```bash
git clone https://github.com/amazon/pytorch-leap
cd pytorch-leap/src/leap
pip install -e .
```

This installs the ``leap`` package and the ``Leap`` meta-learner class. The meta-learner can be used with any
``torch.nn.Module`` class as follows:

```python
Require: criterion, model, tasks, opt_cls, meta_opt_cls, opt_kwargs, meta_opt_kwargs

leap = Leap(model)
mopt = meta_opt_cls(leap.parameters(), **meta_opt_kwargs)
for meta_steps:
    meta_batch = tasks.sample()
    for task in meta_batch:
        leap.init_task()
        leap.to(model)
        opt = opt_cls(model.parameters(), **opt_kwargs)

        for x, y in task:
            loss = criterion(model(x), y)
            loss.backward()

            leap.update(loss, model)

            opt.step()
            opt.zero_grad()  # MUST come after leap.update
    ###
    leap.normalize()
    meta_optimizer.step()
    meta_optimizer.zero_grad()
```

## Omniglot

To run the Omniglot experiment, first prepare the dataset using the ``make_omniglot.sh`` script
in the root directory. The ``p`` flag downloads the dataset, ``d`` installs dependencies and ``l``
creates log directories.

```bash
bash make_omniglot.sh -pdl
```

To train a meta-learner, use the ``main.py`` script. To replicate experiments in the paper select a meta
learner and number of pretraining tasks. For instance, to train ``Leap`` using ``20``
meta-training tasks, execute

```bash
python main.py --meta_model leap --num_pretrain 20 --suffix myrun
```

Logged results can be inspected and visualised using the ``monitor.FileHandler`` class. For all runtime options see

```bash
python main.py -h
```

Meta-learners available:

- ``leap`` (requires the ``src/leap`` package)
- ``reptile``
- ``fomaml``
- ``maml`` (requires the ``src/maml`` package)
- ``ft`` (multi-headed finetuning)
- ``no`` (no meta-training)

