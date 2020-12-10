import ray

import numpy as np
import datetime
import copy

from collections import deque

from core.population import OnePlusOnePoluation, AesrlPopulation, ACemrlPopulation
from core.manager import Manager
from utils.reporter import TrainReporterClock
from utils.color_console import prValuedColor, prColor
from utils.util import get_env_min_max
from .common import Algorithm


class AesrlAlgorithm(Algorithm):
    def __init__(self, args):
        super(AesrlAlgorithm, self).__init__(args)

        self.max_timestep = args.max_timestep
        self.initial_steps = args.initial_steps

        self.population_size = args.population_size
        self.sigma_init = args.sigma_init
        self.algorithm = args.algorithm
        self.rl_ratio = args.aesrl_rl_ratio
        self.rl_k = args.aesrl_rl_k
        self.critic_batch = args.aesrl_critic_batch
        self.critic_worker_num = args.num_critic_worker
        self.actor_worker_num = args.num_actor_worker

        self.args = args

    def get_log_folder_name(self):
        if self.algorithm == 'AES-RL':
            mean_update_str = {'fixed-linear': 'L', 'fixed-sigmoid': 'S',
                               'baseline-absolute': 'A', 'baseline-relative': 'R'}
            var_update_str = {'fixed': 'F', 'adaptive': 'A'}
            str1 = f'AES_{mean_update_str[self.args.aesrl_mean_update]}{var_update_str[self.args.aesrl_var_update]}_' \
                   f'{int(self.args.aesrl_mean_update_param)}'

        elif self.algorithm == '(1+1)-ES':
            str1 = f'OPO'
        elif self.algorithm == 'ACEM-RL':
            str1 = f'ACEM_{self.rl_ratio:.2f}_{self.population_size}'
        else:
            raise NotImplementedError
        return str1

    def assign_workers(self, critic_workers, actor_workers):
        self.critic_workers = critic_workers
        self.actor_workers = actor_workers
        if self.args.actor_network_scheme == 'TD3':
            self.individual_dim = ray.get(self.actor_workers[0].count_parameters.remote(name='main.actor'))
        else:
            raise NotImplementedError
        mean = ray.get(self.actor_workers[0].get_weight.remote(name='main.actor'))

        if self.algorithm == '(1+1)-ES':
            self.population = OnePlusOnePoluation(self.individual_dim, mean, self.args)
        elif self.algorithm == 'AES-RL':
            self.population = AesrlPopulation(self.individual_dim, mean, self.args)
        elif self.algorithm == 'ACEM-RL':
            self.population = ACemrlPopulation(self.population_size, self.individual_dim, mean, self.args)
        else:
            raise NotImplementedError

    def learn(self):
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

        total_steps = 0
        episode_count = 0
        eval_steps = 0
        last_steps = deque(maxlen=10)
        last_results = []
        critic_steps = 0

        total_rl_population = 1
        total_es_population = 1

        training_kwargs = copy.copy(ray.get(self.actor_workers[0].get_default_training_kwargs.remote()))

        ray.get([worker.set_eval.remote() for worker in self.actor_workers])
        ray.get([worker.set_train.remote() for worker in self.critic_workers])

        max_reward = -10000
        eval_max_reward = -10000
        init_time = datetime.datetime.now()
        num_evals = 5

        env_min, env_max = get_env_min_max(self.args.env_name)

        while True:
            if total_steps >= self.max_timestep and manager.num_running_worker('actor_worker') == 0:
                break

            # critic
            if total_steps >= self.initial_steps:
                if critic_steps < total_steps < self.max_timestep:
                    while manager.num_idle_worker('critic_worker') > 0:
                        worker, worker_idx = manager.get_idle_worker(name='critic_worker')
                        ray.get(worker.set_weight.remote(shared_critic, name=critic_names))
                        mean_weight, _ = self.population.get_mean()

                        ray.get(worker.set_weight.remote(mean_weight, name='main.actor'))
                        ray.get(worker.set_weight.remote(mean_weight, name='target.actor'))

                        training_kwargs['learn_critic'] = True
                        training_kwargs['learn_actor'] = False
                        training_kwargs['reset_optim'] = False
                        training_kwargs['batches'] = self.critic_batch
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
                    critic_steps += self.critic_batch

                    shared_critic = trained_critic

            # actor
            if total_steps < self.max_timestep:
                if manager.num_idle_worker('actor_worker') > 0:
                    worker, worker_idx = manager.get_idle_worker('actor_worker')

                    individual = self.population.ask(1)

                    worker.set_weight.remote(individual, name='main.actor')
                    if total_steps < self.args.initial_steps:
                        rl_prob = 0.0
                    else:
                        current_rl_ratio = total_rl_population / (total_rl_population + total_es_population)
                        rl_prob = -self.rl_k * (current_rl_ratio - self.rl_ratio) + 0.5

                    # es_rpob = 1 - rl_prob
                    if np.random.rand(1) <= rl_prob:
                        total_rl_population += 1
                        worker.set_weight.remote(shared_critic, name=critic_names)
                        worker.set_weight.remote(individual, name='target.actor')
                        ray.get(worker.set_train.remote())

                        training_kwargs['learn_critic'] = False
                        training_kwargs['learn_actor'] = True
                        training_kwargs['reset_optim'] = True
                        training_kwargs['batches'] = sum(last_steps)

                        manager.new_job('train',
                                        job_name='actor_worker',
                                        **training_kwargs)
                    else:
                        if total_steps >= self.args.initial_steps:
                            total_es_population += 1
                        random_action = False if total_steps >= self.args.initial_steps else True
                        manager.new_job('rollout', job_name='actor_worker',
                                        job_setting={'individual': individual, 'eval': 'individual'},
                                        random_action=random_action, eval=False, mid_train=False, noise=False)

            result = manager.wait(name='actor_worker', remove=False, timeout=0)
            if result is not None:
                finished_job, finished_job_id, finished_worker_dict = result
                finished_worker = finished_worker_dict['worker']

                if finished_worker_dict['func_name'] == 'train':
                    random_action = False if total_steps >= self.args.initial_steps else True
                    ray.get(finished_worker.set_eval.remote())

                    individual = ray.get(finished_worker.get_weight.remote(name='main.actor'))

                    manager.done(finished_worker_dict)

                    manager.new_job('rollout', job_name='actor_worker',
                                    specific_worker=finished_worker,
                                    job_setting={'individual': individual, 'eval': 'individual'},
                                    random_action=random_action, eval=False, mid_train=False, noise=False)

                elif finished_worker_dict['func_name'] == 'rollout':
                    episode_t, episode_reward = ray.get(finished_job_id)

                    eval_type = finished_worker_dict['setting']['eval']
                    if eval_type == 'individual':

                        total_steps += episode_t
                        last_steps.append(episode_t)
                        eval_steps += episode_t
                        episode_count += 1

                        individual = finished_worker_dict['setting']['individual']

                        mean_weight, _ = self.population.get_mean()

                        ray.get(finished_worker.set_weight.remote(mean_weight, name='main.actor'))

                        manager.done(finished_worker_dict)

                        self.summary.add_scalar('train/individuals', episode_reward, total_steps)

                        manager.new_job('rollout', job_name='actor_worker',
                                        specific_worker=finished_worker,
                                        job_setting={'individual': individual,
                                                     'eval': 'mean',
                                                     'step': total_steps,
                                                     'individual_reward': episode_reward,
                                                     'eval_counter': 0,
                                                     'mean_reward': []},
                                        random_action=False, eval=True, mid_train=False, noise=False)
                    elif eval_type == 'mean':
                        step = finished_worker_dict['setting']['step']
                        individual_reward = finished_worker_dict['setting']['individual_reward']
                        individual = finished_worker_dict['setting']['individual']
                        eval_counter = finished_worker_dict['setting']['eval_counter'] + 1
                        mean_reward = finished_worker_dict['setting']['mean_reward']

                        mean_t, cur_mean_reward = ray.get(finished_job_id)
                        mean_reward.append(cur_mean_reward)

                        if eval_counter >= num_evals:
                            mean_reward = float(np.average(np.array(mean_reward)))
                            self.population.update_mean_result(mean_reward)

                            p = self.population.tell(individual, individual_reward)
                            # var = self.population.var[0]

                            self.summary.add_scalar('test/mean', mean_reward, step)
                            if mean_reward > eval_max_reward:
                                eval_max_reward = mean_reward
                                self.summary.add_scalar('test/max', eval_max_reward, step)

                            if individual_reward > max_reward:
                                max_reward = individual_reward
                                self.summary.add_scalar('train/max', max_reward, step)

                            self.summary.add_scalar('train/p', p, step)

                            elapsed = (datetime.datetime.now() - init_time).total_seconds()

                            self.logger.log(f'Total step: {step}, time: {elapsed:.2f} s, '
                                            f'max_reward: ' +
                                            prColor(f'{max_reward:.3f}',
                                                    fore=prValuedColor(max_reward, env_min, env_max, 40,
                                                                       "#600000", "#00F0F0")) +
                                            f', '
                                            f'individual_results: ' +
                                            prColor(f'{individual_reward:.3f}',
                                                    fore=prValuedColor(individual_reward, env_min, env_max, 40,
                                                                       "#600000", "#00F0F0")) +
                                            f', '
                                            f'critic_steps: {critic_steps}, ' +
                                            f'mean_reward: ' +
                                            prColor(f'{mean_reward:.3f}',
                                                    fore=prValuedColor(mean_reward, env_min, env_max, 40,
                                                                       "#600000", "#00F0F0")) +
                                            f', p: {p:.3f}, '
                                            # f'var: {var:.6f}, '
                                            f'rl: {total_rl_population}, es: {total_es_population}')
                            manager.done(finished_worker_dict)
                        else:
                            manager.done(finished_worker_dict)
                            manager.new_job('rollout', job_name='actor_worker',
                                            specific_worker=finished_worker,
                                            job_setting={'individual': individual,
                                                         'eval': 'mean',
                                                         'step': step,
                                                         'individual_reward': individual_reward,
                                                         'eval_counter': eval_counter,
                                                         'mean_reward': mean_reward},
                                            random_action=False, eval=True, mid_train=False, noise=False)

                    else:
                        raise NotImplementedError

                else:
                    raise NotImplementedError

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

