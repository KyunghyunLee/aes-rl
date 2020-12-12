import ray

import numpy as np
import datetime
import copy

from collections import deque

from core.population import CemrlPopulation
from core.manager import Manager
from utils.reporter import TrainReporterClock
from .common import Algorithm
from utils.color_console import prColor, prValuedColor
from utils.util import get_env_min_max


class PCemrlAlgorithm(Algorithm):
    def __init__(self, args):
        super(PCemrlAlgorithm, self).__init__(args)

        self.max_timestep = args.max_timestep
        self.initial_steps = args.initial_steps

        self.population_size = args.population_size
        self.sigma_init = args.sigma_init
        self.damp_init = args.cemrl_damp_init
        self.damp_limit = args.cemrl_damp_limit
        self.damp_tau = args.cemrl_damp_tau
        self.antithetic = not args.cemrl_no_antithetic
        self.rl_population = args.cemrl_rl_population
        self.parallel_critic = args.parallel_critic

        self.algorithm = args.algorithm

        self.critic_worker_num = args.num_critic_worker
        self.actor_worker_num = args.num_actor_worker

    def get_log_folder_name(self):
        if self.parallel_critic:
            parallel_str = 'P'
        else:
            parallel_str = 'S'
        str1 = f'PCEM_{parallel_str}_{self.rl_population}_{self.population_size}'
        return str1

    def assign_workers(self, critic_workers, actor_workers):
        self.critic_workers = critic_workers
        self.actor_workers = actor_workers
        if self.args.actor_network_scheme == 'TD3':
            self.individual_dim = ray.get(self.actor_workers[0].count_parameters.remote(name='main.actor'))
        else:
            raise NotImplementedError
        mean = ray.get(self.actor_workers[0].get_weight.remote(name='main.actor'))
        self.population = CemrlPopulation(self.population_size, self.individual_dim, mean, self.args)

    def learn(self):
        if self.parallel_critic:
            self.learn_parallel_critic()
        else:
            self.learn_serial_critic()

    def learn_serial_critic(self):
        manager = Manager(self.logger)
        # report_clock = TrainReporterClock.remote()
        # manager.add_worker(report_clock, 'report_timeout')

        for worker in self.actor_workers:
            manager.add_worker(worker, 'actor_worker')

        training_kwargs = copy.copy(ray.get(self.actor_workers[0].get_default_training_kwargs.remote()))

        algorithm_state = 'population_ask'

        total_steps = 0
        episode_count = 0
        prev_actor_steps = 0
        eval_steps = 0
        current_actor_step = 0
        max_reward = -10000
        eval_max_reward = -10000

        individuals = []
        results = []
        individuals_queue = deque(maxlen=self.population_size)

        critic_training_index = 0

        ray.get([worker.set_eval.remote() for worker in self.critic_workers + self.actor_workers])
        critic_names = ray.get(self.actor_workers[0].get_critic_names.remote())
        init_time = datetime.datetime.now()
        env_min, env_max = get_env_min_max(self.args.env_name)

        while True:
            if total_steps >= self.max_timestep and manager.num_running_worker('actor_worker') == 0 \
                    and algorithm_state == 'population_ask':
                break

            if algorithm_state == 'population_ask':
                individuals = self.population.ask(self.population_size)
                results = [None for _ in range(self.population_size)]
                if total_steps >= self.args.initial_steps:
                    algorithm_state = 'critic_training'
                else:
                    for idx in range(self.population_size):
                        individuals_queue.append(idx)
                    algorithm_state = 'actor_evaluating'
                critic_training_index = 0
                current_actor_step = 0

            if algorithm_state == 'critic_training':
                if manager.get_worker_state_by_index('actor_worker', 0) == 'idle':
                    worker = manager.get_worker_by_index('actor_worker', 0)
                    worker.set_train.remote()

                    worker.set_weight.remote(individuals[critic_training_index], name='main.actor')
                    worker.set_weight.remote(individuals[critic_training_index], name='target.actor')

                    training_kwargs['learn_critic'] = True
                    training_kwargs['learn_actor'] = False
                    training_kwargs['reset_optim'] = False
                    training_kwargs['batches'] = int(prev_actor_steps / self.rl_population)

                    manager.new_job('train', specific_worker=worker,
                                    job_name='actor_worker', job_setting=None, **training_kwargs)

                result = manager.wait(name='actor_worker', remove=False)
                if result is not None:
                    finished_job, finished_job_id, finished_worker_dict = result
                    finished_worker = finished_worker_dict['worker']
                    finished_worker.set_eval.remote()

                    critic_training_index += 1
                    manager.done(finished_worker_dict)

                    if critic_training_index >= self.rl_population:
                        for idx in range(self.rl_population):
                            individuals_queue.append(idx)

                        critic_weight = ray.get(finished_worker.get_weight.remote(name=critic_names))
                        critic_weight_obj = ray.put(critic_weight)
                        set_critic_weight_obj = [actor_worker.set_weight.remote(critic_weight_obj, name=critic_names)
                                                 for actor_worker in self.actor_workers]
                        ray.get(set_critic_weight_obj)
                        algorithm_state = 'actor_training'
                        set_train_obj = [worker.set_train.remote() for worker in self.actor_workers]
                        ray.get(set_train_obj)

            if algorithm_state == 'actor_training':
                if len(individuals_queue) == 0 and manager.num_running_worker('actor_worker') == 0:
                    algorithm_state = 'actor_evaluating'
                    set_train_obj = [worker.set_eval.remote() for worker in self.actor_workers]
                    ray.get(set_train_obj)

                    for idx in range(self.population_size):
                        individuals_queue.append(idx)

                elif manager.num_idle_worker('actor_worker') > 0 and len(individuals_queue) > 0:
                    individual_idx = individuals_queue.popleft()
                    worker, worker_idx = manager.get_idle_worker('actor_worker')
                    worker.set_weight.remote(individuals[individual_idx], name='main.actor')
                    worker.set_weight.remote(individuals[individual_idx], name='target.actor')
                    ray.get(worker.set_train.remote())

                    training_kwargs['learn_critic'] = False
                    training_kwargs['learn_actor'] = True
                    training_kwargs['reset_optim'] = True
                    training_kwargs['batches'] = int(prev_actor_steps)
                    training_kwargs['individual_id'] = individual_idx
                    manager.new_job('train',
                                    job_name='actor_worker', job_setting={'individual_idx': individual_idx},
                                    **training_kwargs)

                result = manager.wait(name='actor_worker', remove=False, timeout=0)
                if result is not None:
                    finished_job, finished_job_id, finished_worker_dict = result
                    finished_worker = finished_worker_dict['worker']

                    finished_individual = finished_worker_dict['setting']['individual_idx']
                    trained_weight = ray.get(finished_worker.get_weight.remote(name='main.actor'))
                    individuals[finished_individual] = trained_weight
                    manager.done(finished_worker_dict)

            if algorithm_state == 'actor_evaluating':
                if len(individuals_queue) == 0 and manager.num_running_worker('actor_worker') == 0:
                    algorithm_state = 'population_tell'

                elif manager.num_idle_worker('actor_worker') > 0 and len(individuals_queue) > 0:
                    individual_idx = individuals_queue.popleft()
                    worker, worker_idx = manager.get_idle_worker('actor_worker')
                    worker.set_weight.remote(individuals[individual_idx], name='main.actor')
                    # worker.set_weight.remote(individuals[individual_idx], name='target.actor')

                    random_action = False if total_steps >= self.args.initial_steps else True

                    manager.new_job('rollout', job_name='actor_worker', job_setting={'individual_idx': individual_idx},
                                    random_action=random_action, eval=False, mid_train=False)

                result = manager.wait(name='actor_worker', remove=False, timeout=0)
                if result is not None:
                    finished_job, finished_job_id, finished_worker_dict = result
                    finished_worker = finished_worker_dict['worker']

                    finished_individual = finished_worker_dict['setting']['individual_idx']
                    episode_t, episode_reward = ray.get(finished_job_id)
                    results[finished_individual] = episode_reward

                    manager.done(finished_worker_dict)

                    total_steps += episode_t
                    current_actor_step += episode_t
                    eval_steps += episode_t
                    episode_count += 1

                    self.summary.add_scalar('train/individuals', episode_reward, total_steps)
                    if episode_reward > max_reward:
                        max_reward = episode_reward
                        self.summary.add_scalar('train/max', max_reward, total_steps)

            if algorithm_state == 'population_tell':
                self.population.tell(individuals, results)
                elapsed = (datetime.datetime.now() - init_time).total_seconds()
                result_str = [prColor(f'{result:.2f}',
                                      fore=prValuedColor(result, env_min, env_max, 40, "#600000", "#00F0F0"))
                              for result in results]

                result_str = ', '.join(result_str)

                self.logger.log(f'Total step: {total_steps}, time: {elapsed:.2f} s, '
                                f'max_reward: ' +
                                prColor(f'{max_reward:.3f}',
                                        fore=prValuedColor(max_reward, env_min, env_max, 40, "#600000", "#00F0F0")) +
                                f', results: {result_str}')

                prev_actor_steps = current_actor_step
                algorithm_state = 'mean_eval'
                # algorithm_state = 'population_ask'

            if algorithm_state == 'mean_eval':
                mean_weight, var_weight = self.population.get_mean()
                worker, worker_idx = manager.get_idle_worker('actor_worker')
                ray.get(worker.set_weight.remote(mean_weight, name='main.actor'))
                manager.new_job('rollout', job_name='actor_worker', job_setting=None,
                                random_action=False, eval=True, mid_train=False)
                result = manager.wait(name='actor_worker', remove=False, timeout=None)
                if result is not None:
                    finished_job, finished_job_id, finished_worker_dict = result
                    finished_worker = finished_worker_dict['worker']

                    eval_t, eval_reward = ray.get(finished_job_id)
                    manager.done(finished_worker_dict)

                    if eval_reward > eval_max_reward:
                        eval_max_reward = eval_reward
                        self.summary.add_scalar('test/max', eval_reward, total_steps)

                    self.summary.add_scalar('test/mu', eval_reward, total_steps)

                    algorithm_state = 'population_ask'

            if eval_steps >= 50000:
                eval_t, eval_reward = self.run()
                self.logger.log(f'Evaluation: {eval_reward}')
                eval_steps = 0

    def learn_parallel_critic(self):
        manager = Manager(self.logger)
        for worker in self.actor_workers:
            manager.add_worker(worker, 'actor_worker')

        for worker in self.critic_workers:
            manager.add_worker(worker, 'critic_worker')

        critic_names = ray.get(self.critic_workers[0].get_critic_names.remote())

        shared_critic = ray.get(self.critic_workers[0].get_weight.remote(name=critic_names))
        shared_critic_obj = ray.put(shared_critic)
        critic_sync_obj = ray.get([worker.set_weight.remote(shared_critic_obj, name=critic_names)
                                   for worker in self.critic_workers])
        ray.get([worker.set_train.remote() for worker in self.critic_workers])

        training_kwargs = copy.copy(ray.get(self.actor_workers[0].get_default_training_kwargs.remote()))

        total_steps = 0
        episode_count = 0
        prev_actor_steps = 0
        eval_steps = 0
        current_actor_step = 0
        max_reward = -10000
        eval_max_reward = -10000

        individuals = []
        results = []
        individuals_queue = deque(maxlen=self.population_size)

        ray.get([worker.set_eval.remote() for worker in self.actor_workers])
        ray.get([worker.set_train.remote() for worker in self.critic_workers])

        init_time = datetime.datetime.now()

        algorithm_state = 'population_ask'

        critic_steps = 0

        env_min, env_max = get_env_min_max(self.args.env_name)

        while True:
            if total_steps >= self.max_timestep and manager.num_running_worker(['actor_worker', 'critic_worker']) == 0 \
                    and algorithm_state == 'population_ask':
                break

            if total_steps >= self.initial_steps:
                if total_steps < self.max_timestep:
                    while manager.num_idle_worker('critic_worker') > 0:
                        worker, worker_idx = manager.get_idle_worker(name='critic_worker')
                        ray.get(worker.set_weight.remote(shared_critic, name=critic_names))
                        mean_weight, _ = self.population.get_mean()

                        ray.get(worker.set_weight.remote(mean_weight, name='main.actor'))
                        ray.get(worker.set_weight.remote(mean_weight, name='target.actor'))

                        training_kwargs['learn_critic'] = True
                        training_kwargs['learn_actor'] = False
                        training_kwargs['reset_optim'] = False
                        training_kwargs['batches'] = 1000
                        prev_critic = copy.deepcopy(shared_critic)
                        manager.new_job('train', specific_worker=worker,
                                        job_name='critic_worker', job_setting={'prev_critic': prev_critic},
                                        **training_kwargs)

                result = manager.wait(name='critic_worker', remove=False, timeout=0)
                if result is not None:
                    finished_job, finished_job_id, finished_worker_dict = result
                    finished_worker = finished_worker_dict['worker']
                    trained_critic = ray.get(finished_worker.get_weight.remote(name=critic_names))
                    prev_critic = finished_worker_dict['setting']['prev_critic']

                    manager.done(finished_worker_dict)
                    critic_steps += 1000
                    if self.critic_worker_num > 1:
                        d_critic = trained_critic - prev_critic

                        shared_critic = shared_critic + d_critic
                    else:
                        shared_critic = trained_critic

            if algorithm_state == 'population_ask':
                individuals = self.population.ask(self.population_size)
                results = [None for _ in range(self.population_size)]
                if total_steps < self.args.initial_steps:
                    for idx in range(self.population_size):
                        individuals_queue.append(idx)
                    algorithm_state = 'actor_evaluating'
                else:
                    for idx in range(self.rl_population):
                        individuals_queue.append(idx)
                    algorithm_state = 'actor_training'
                current_actor_step = 0

            if algorithm_state == 'actor_training':
                if len(individuals_queue) == 0 and manager.num_running_worker('actor_worker') == 0:
                    algorithm_state = 'actor_evaluating'
                    set_train_obj = [worker.set_eval.remote() for worker in self.actor_workers]
                    ray.get(set_train_obj)

                    for idx in range(self.population_size):
                        individuals_queue.append(idx)

                elif manager.num_idle_worker('actor_worker') > 0 and len(individuals_queue) > 0:
                    individual_idx = individuals_queue.popleft()
                    worker, worker_idx = manager.get_idle_worker('actor_worker')

                    worker.set_weight.remote(shared_critic, name=critic_names)

                    worker.set_weight.remote(individuals[individual_idx], name='main.actor')
                    worker.set_weight.remote(individuals[individual_idx], name='target.actor')
                    ray.get(worker.set_train.remote())

                    training_kwargs['learn_critic'] = False
                    training_kwargs['learn_actor'] = True
                    training_kwargs['reset_optim'] = True
                    training_kwargs['batches'] = int(prev_actor_steps)
                    training_kwargs['individual_id'] = individual_idx
                    manager.new_job('train',
                                    job_name='actor_worker', job_setting={'individual_idx': individual_idx},
                                    **training_kwargs)

                result = manager.wait(name='actor_worker', remove=False, timeout=0)
                if result is not None:
                    finished_job, finished_job_id, finished_worker_dict = result
                    finished_worker = finished_worker_dict['worker']

                    finished_individual = finished_worker_dict['setting']['individual_idx']
                    trained_weight = ray.get(finished_worker.get_weight.remote(name='main.actor'))
                    individuals[finished_individual] = trained_weight
                    manager.done(finished_worker_dict)

            if algorithm_state == 'actor_evaluating':
                if len(individuals_queue) == 0 and manager.num_running_worker('actor_worker') == 0:
                    algorithm_state = 'population_tell'

                elif manager.num_idle_worker('actor_worker') > 0 and len(individuals_queue) > 0:
                    individual_idx = individuals_queue.popleft()
                    worker, worker_idx = manager.get_idle_worker('actor_worker')
                    worker.set_weight.remote(individuals[individual_idx], name='main.actor')
                    # worker.set_weight.remote(individuals[individual_idx], name='target.actor')

                    random_action = False if total_steps >= self.args.initial_steps else True

                    manager.new_job('rollout', job_name='actor_worker', job_setting={'individual_idx': individual_idx},
                                    random_action=random_action, eval=False, mid_train=False)

                result = manager.wait(name='actor_worker', remove=False, timeout=0)
                if result is not None:
                    finished_job, finished_job_id, finished_worker_dict = result
                    finished_worker = finished_worker_dict['worker']

                    finished_individual = finished_worker_dict['setting']['individual_idx']
                    episode_t, episode_reward = ray.get(finished_job_id)
                    results[finished_individual] = episode_reward

                    manager.done(finished_worker_dict)

                    total_steps += episode_t
                    current_actor_step += episode_t
                    eval_steps += episode_t
                    episode_count += 1

                    self.summary.add_scalar('train/individuals', episode_reward, total_steps)
                    if episode_reward > max_reward:
                        max_reward = episode_reward
                        self.summary.add_scalar('train/max', max_reward, total_steps)

            if algorithm_state == 'population_tell':
                self.population.tell(individuals, results)
                elapsed = (datetime.datetime.now() - init_time).total_seconds()
                result_str = [prColor(f'{result:.2f}',
                                      fore=prValuedColor(result, env_min, env_max, 40, "#600000", "#00F0F0"))
                              for result in results]

                result_str = ', '.join(result_str)

                self.logger.log(f'Total step: {total_steps}, time: {elapsed:.2f} s, '
                                f'max_reward: ' +
                                prColor(f'{max_reward:.3f}',
                                        fore=prValuedColor(max_reward, env_min, env_max, 40, "#600000", "#00F0F0")) +
                                f', results: {result_str}, '
                                f'critic_steps: {critic_steps}')

                prev_actor_steps = current_actor_step
                algorithm_state = 'mean_eval'
                # algorithm_state = 'population_ask'

            if algorithm_state == 'mean_eval':
                mean_weight, var_weight = self.population.get_mean()
                worker, worker_idx = manager.get_idle_worker('actor_worker')
                ray.get(worker.set_weight.remote(mean_weight, name='main.actor'))
                manager.new_job('rollout', job_name='actor_worker', job_setting=None,
                                random_action=False, eval=True, mid_train=False)
                result = manager.wait(name='actor_worker', remove=False, timeout=None)
                if result is not None:
                    finished_job, finished_job_id, finished_worker_dict = result
                    finished_worker = finished_worker_dict['worker']

                    eval_t, eval_reward = ray.get(finished_job_id)
                    manager.done(finished_worker_dict)

                    if eval_reward > eval_max_reward:
                        eval_max_reward = eval_reward
                        self.summary.add_scalar('test/max', eval_reward, total_steps)

                    self.summary.add_scalar('test/mean', eval_reward, total_steps)

                    algorithm_state = 'population_ask'

            if eval_steps >= 50000:
                eval_t, eval_reward = self.run()
                self.logger.log(f'Evaluation: {eval_reward}')
                eval_steps = 0

    def run(self, count=1):
        mean, _ = self.population.get_mean()
        ray.get(self.actor_workers[0].set_weight.remote(mean, name='main.actor'))
        results = []
        result = None
        for idx in range(count):
            result = ray.get(self.actor_workers[0].rollout.remote(
                random_action=False, eval=True, mid_train=False))
            results.append(result)

        if count == 1:
            return result

        results = zip(*results)
        return results

