"""
Official training code for a NIPS 2020 Oral paper,
An Efficient Asynchronous Method for Integrating Evolutionary and Gradient-based Policy Search

Author: Kyunghyun Lee
Email: kyunghyun.lee@kaist.ac.kr
"""

import argparse
import datetime
import os
import json

import ray

import git
import GPUtil

from torch.utils.tensorboard import SummaryWriter

from utils.util import init_argument_checker, init_argument_parser, init_gym_from_args
from utils.logger import Printer, Logger
from utils.color_console import prAuto
from utils.git_sync import GitWorker

from core.worker import Worker
from core.replay import ReplayBuffer

from algorithm import Td3Algorithm, CemrlAlgorithm, PCemrlAlgorithm, AesrlAlgorithm

from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group
)


def init_logger(args, algorithm):
    algorithm_dir = algorithm.get_log_folder_name()
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    algorithm_path = f'{algorithm_dir}_{time_str}'
    logger_path = os.path.join(args.logdir, f'{args.env_name}', algorithm_path)
    # logger_path = os.path.join(os.path.join(f'{args.env_name}', args.logdir), time_str)
    logger = Logger(path=logger_path) if args.logdir != '' else Printer()
    summary_writer = SummaryWriter(log_dir=logger_path)
    return logger, summary_writer


def init_ray(args, logger):
    if args.debug:
        logger.log(prAuto('[INFO] Run in local mode'))
        ray.init(local_mode=True, ignore_reinit_error=True, resources={'head': 1, 'machine': 1})

    elif args.ray_address is None:  # Auto mode in a single machine
        ray.init(ignore_reinit_error=True, resources={'head': 1, 'machine': 1})

    else:  #
        ray_address = f'{args.ray_address}:{args.ray_port}'
        ray.init(address=ray_address, _redis_password=args.redis_password, ignore_reinit_error=True)

    resources = ray.available_resources()
    total_gpu = int(resources['GPU'])

    if resources['machine'] >= 2:
        # todo: confirm
        GitWorkerHead = ray.remote(num_cpus=1, resources={'machine': 1, 'head': 0.01})(GitWorker)
        GitWorkerOther = ray.remote(num_cpus=1, resources={'machine': 1})(GitWorker)

        git_workers = [GitWorkerHead.remote()]
        git_workers += [GitWorkerOther.remote() for _ in range(int(resources['machine']) - 1)]

        all_gpu = ray.get([worker.get_gpu_count.remote() for worker in git_workers])

        repo = git.Repo(os.path.dirname(os.path.abspath(__file__)))
        if repo.is_dirty():
            logger.log(prAuto('[WARNING] Master Git is dirty! Remote can cause undesired running'))
        else:
            logger.log(prAuto('[INFO] Git clean'))
            master_sha = repo.head.object.hexsha
            ray.get([worker.sync.remote(master_sha) for worker in git_workers])
        for worker in git_workers:
            ray.kill(worker)
    else:
        all_gpu = [total_gpu]
    return total_gpu, all_gpu


def init_replay(args):
    # todo: init replay in multiple machines and sync.
    env, state_dim, action_dim = init_gym_from_args(args)
    env.close()
    replay = ReplayBuffer.remote(state_dim, action_dim, args.replay_size)
    return replay


def init_workers(args, algorithm, logger, replay, total_gpu, reporter, all_gpu):
    # worker_index
    #   [0, num_critic_worker) : critic workers
    #   [num_critic_worker, total_workers) : actor workers

    total_critic_worker, total_actor_worker = algorithm.get_workers()

    total_worker = total_critic_worker + total_actor_worker
    gpu_assign = [[] for _ in range(total_gpu)]
    worker_gpu_num = [0 for _ in range(total_worker)]
    cur_idx = 0

    if total_critic_worker > 1:
        print(prAuto('[ERROR] Multiple critic workers are currently not supported'))
        exit(0)

    for worker_idx in range(total_critic_worker):
        gpu_assign[cur_idx].append(worker_idx)
        worker_gpu_num[worker_idx] = cur_idx
        cur_idx = (cur_idx + 1) % total_gpu

    first = True

    for worker_idx in range(total_actor_worker):
        actor_worker_idx = worker_idx + total_critic_worker
        gpu_assign[cur_idx].append(actor_worker_idx)
        cur_idx = cur_idx + 1
        if cur_idx >= total_gpu:
            cur_idx = 0 if not first else total_critic_worker % total_gpu
            first = False

    critic_workers = []
    actor_workers = []

    worker_limit = 3
    worker_num_in_gpu = [len(gpu_assign[gpu_idx]) for gpu_idx in range(total_gpu)]
    worker_limit_list = [worker_num_in_gpu[gpu_idx] > worker_limit for gpu_idx in range(total_gpu)]

    if any(worker_limit_list):
        logger.log(prAuto(f'[WARNING] Assigning more than {worker_limit} workers'
                          f' (max: {max(worker_num_in_gpu)}) in one GPU. '
                          f'The learning process can be slowed'))
    critic_idx = 0
    actor_idx = 0

    zipped = zip(worker_num_in_gpu, gpu_assign)
    zipped = sorted(zipped, reverse=True)
    gpu_assign = [element for _, element in zipped]

    param_groups = []

    for gpu_idx in range(total_gpu):
        worker_in_gpu = len(gpu_assign[gpu_idx])
        if worker_in_gpu == 0:
            continue

        if gpu_idx < all_gpu[0]:
            pg = placement_group([{'GPU': 1, 'CPU': worker_in_gpu, 'head': 0.1}], strategy='STRICT_PACK')
        else:
            pg = placement_group([{'GPU': 1, 'CPU': worker_in_gpu}], strategy='STRICT_PACK')
        ray.get(pg.ready())
        param_groups.append(pg)

    for gpu_idx in range(total_gpu):
        worker_in_gpu = len(gpu_assign[gpu_idx])
        if worker_in_gpu == 0:
            continue

        if gpu_idx < all_gpu[0]:
            WorkerGPU = ray.remote(num_cpus=1, num_gpus=1.0 / worker_in_gpu, resources={'head': 0.001})(Worker)
        else:
            WorkerGPU = ray.remote(num_cpus=1, num_gpus=1.0 / worker_in_gpu)(Worker)
        for worker_idx in gpu_assign[gpu_idx]:
            if 0 <= worker_idx < total_critic_worker:
                critic_workers.append(WorkerGPU.options(name=f'critic_worker_{critic_idx}',
                                                        placement_group=param_groups[gpu_idx]
                                                        )
                                      .remote(replay, f'critic_worker_{critic_idx}', args, logger, reporter))
                critic_idx += 1
            elif worker_idx < total_critic_worker + total_actor_worker:
                actor_workers.append(WorkerGPU.options(name=f'actor_worker_{actor_idx}',
                                                       placement_group=param_groups[gpu_idx])
                                     .remote(replay, f'actor_worker_{actor_idx}', args, logger, reporter))
                actor_idx += 1

    return critic_workers, actor_workers


def init_algorithm(args):
    if args.algorithm == 'TD3':
        algorithm = Td3Algorithm(args)
    elif args.algorithm == 'CEM-RL':
        algorithm = CemrlAlgorithm(args)
    elif args.algorithm == 'P-CEM-RL':
        algorithm = PCemrlAlgorithm(args)
    elif args.algorithm in ['AES-RL', '(1+1)-ES', 'ACEM-RL']:
        algorithm = AesrlAlgorithm(args)
    else:
        raise NotImplementedError
    return algorithm


def main(args):
    algorithm = init_algorithm(args)
    logger, summary = init_logger(args, algorithm)
    algorithm.set_logger(logger, summary)

    logger.log(json.dumps(vars(args), indent=2))
    total_gpu, all_gpu = init_ray(args, logger)
    replay = init_replay(args)
    algorithm.set_replay(replay)

    critic_workers, actor_workers = init_workers(args, algorithm, logger, replay, total_gpu, None, all_gpu)

    algorithm.assign_workers(critic_workers, actor_workers)

    algorithm.learn()


if __name__ == '__main__':
    args, ok_flag = init_argument_parser()
    argcheck = init_argument_checker()

    if argcheck.check(args) and ok_flag:
        main(args)
