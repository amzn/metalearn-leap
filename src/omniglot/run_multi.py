import os
import time
import argparse
from multiprocessing import Pool
import random

parser = argparse.ArgumentParser(description='Multi-Omniglot experiment')
parser.add_argument('--model', required=True, type=str, help='Meta model to run')
parser.add_argument('--niter', default=10, type=int, help='(default=%(default)d)')
parser.add_argument('--ngpus', default=1, type=int, help='(default=%(default)d)')
parser.add_argument('--npres', default=[1, 3, 5, 10, 15, 20, 25],
                    type=int, nargs='+', help='(default=%(default)r)')
parser.add_argument('--npar', default=2, type=int, help='(default=%(default)d)')
parser.add_argument('--sleep', default=120, type=float, help='(default=%(default)f)')
parser.add_argument('--seed', default=6534, type=int, help='(default=%(default)d)')
args = parser.parse_args()


def get_async(results, pool):
    """Clear completed jobs from results cache"""
    complete = []
    for i, (res, _) in enumerate(results):
        if res.ready():
            complete.append(i)

    for i in reversed(complete):
        res, idx = results.pop(i)

        res.get()
        pool.append(idx)
    return len(complete)


def cleanup(results):
    """Ensure cached jobs are retrieved"""
    print("Terminating jobs...", end="")
    for res, _ in results: res.get()
    print("done.")


def gen_job(npre, inner_lr, outer_lr, train_steps, model, gpx, seed):
    jobname = "{}_{}_{}_{}_EXAMPLE".format(inner_lr, outer_lr, npre, seed)
    call = "python main.py \
            --workers 0 \
            --num_pretrain {} \
            --classes 20 \
            --meta_train_steps 1000 \
            --meta_batch_size 20 \
            --task_batch_size 20 \
            --task_train_steps {} \
            --task_val_steps 100 \
            --log_ival 0 \
            --write_ival 1 \
            --test_ival 20 \
            --inner_kwargs lr {}\
            --outer_kwargs lr {} \
            --suffix {} \
            --device {} \
            --meta_model {} \
            --seed {}".format(npre, train_steps, inner_lr, outer_lr, jobname, gpx, model, seed)
    return call


def get_model_config(model):
    """Returns hyper-parameters for given mode"""
    if model == 'maml':
        return 0.1, 0.5, 5
    if model == 'fomaml':
        return 0.1, 0.5, 100
    return 0.1, 0.1, 100


def run():
    """Job manager"""
    inner_lr, outer_lr, train_steps = get_model_config(args.model)

    random.seed(args.seed)
    seeds = [random.randint(0, 10000) for _ in range(args.niter)]

    results = []
    pool = Pool(args.ngpus * args.npar)
    available = [i % args.ngpus for i in range(args.ngpus * args.npar)]

    ncomplete = 0
    ntot = len(seeds) * len(args.npres)
    for s in seeds:
        for p in args.npres:
            case_running = False
            while not case_running:
                ncomplete += get_async(results, available)
                if available:
                    gpx = available.pop(0)
                    call = gen_job(p, inner_lr, outer_lr, train_steps, args.model, gpx, s)
                    res = pool.apply_async(os.system, [call])
                    results.append((res, gpx))
                    case_running = True
                else:
                    time.sleep(args.sleep)
            print('Launched {} on {} with seed {} (total completed {}/{})'.format(
                p, gpx, s, ncomplete, ntot))
    cleanup(results)


if __name__ == '__main__':
    run()
