import ray

import numpy as np
import datetime
import copy

from core.population import CemrlPopulation
from .common import Algorithm
from utils.color_console import prColor, prValuedColor
from utils.util import get_env_min_max


class CemrlAlgorithm(Algorithm):
    def __init__(self, args):
        super(CemrlAlgorithm, self).__init__(args)

        self.max_timestep = args.max_timestep
        self.initial_steps = args.initial_steps

        self.population_size = args.population_size
        self.sigma_init = args.sigma_init
        self.damp_init = args.cemrl_damp_init
        self.damp_limit = args.cemrl_damp_limit
        self.damp_tau = args.cemrl_damp_tau
        self.antithetic = not args.cemrl_no_antithetic
        self.rl_population = args.cemrl_rl_population

        self.algorithm = args.algorithm

        self.critic_worker_num = 0
        self.actor_worker_num = 1

    def get_log_folder_name(self):
        str1 = f'CEM_{self.rl_population}_{self.population_size}'
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
        total_steps = 0
        episode_count = 0
        prev_actor_steps = 0
        eval_steps = 0

        training_kwargs = copy.copy(ray.get(self.actor_workers[0].get_default_training_kwargs.remote()))

        ray.get([worker.set_eval.remote() for worker in self.critic_workers + self.actor_workers])

        max_reward = -10000
        eval_max_reward = -10000

        init_time = datetime.datetime.now()

        env_min, env_max = get_env_min_max(self.args.env_name)

        while total_steps < self.max_timestep:
            results = []
            individuals = self.population.ask(self.population_size)

            current_actor_step = 0

            for idx in range(self.population_size):
                set_weight_obj = self.actor_workers[0].set_weight.remote(individuals[idx], name='main.actor')
                self.actor_workers[0].set_individual_id.remote(idx)
                if idx < self.rl_population and total_steps > self.initial_steps:
                    set_weight_obj = self.actor_workers[0].set_weight.remote(individuals[idx], name='target.actor')
                    ray.get(set_weight_obj)
                    self.actor_workers[0].set_train.remote()
                    training_kwargs['learn_critic'] = True
                    training_kwargs['learn_actor'] = False
                    training_kwargs['reset_optim'] = False
                    training_kwargs['batches'] = int(prev_actor_steps / self.rl_population)

                    ray.get(set_weight_obj)
                    ray.get(self.actor_workers[0].train.remote(**training_kwargs))

                    training_kwargs['learn_critic'] = False
                    training_kwargs['learn_actor'] = True
                    training_kwargs['reset_optim'] = True
                    training_kwargs['batches'] = int(prev_actor_steps)
                    ray.get(self.actor_workers[0].train.remote(**training_kwargs))

                    trained_weight = ray.get(self.actor_workers[0].get_weight.remote(name='main.actor'))
                    individuals[idx] = trained_weight
                    self.actor_workers[0].set_eval.remote()

                if total_steps > self.initial_steps:
                    rollout_obj = self.actor_workers[0].rollout.remote(random_action=False, eval=False, mid_train=False)
                else:
                    rollout_obj = self.actor_workers[0].rollout.remote(random_action=True, eval=False, mid_train=False)

                episode_t, episode_reward = ray.get(rollout_obj)

                total_steps += episode_t
                current_actor_step += episode_t
                eval_steps += episode_t

                self.summary.add_scalar('train/individuals', episode_reward, total_steps)

                results.append(episode_reward)
                episode_count += 1
                if episode_reward > max_reward:
                    max_reward = episode_reward
                    self.summary.add_scalar('train/max', episode_reward, total_steps)

            self.population.tell(individuals, results)
            # change to reporter
            elapsed = (datetime.datetime.now() - init_time).total_seconds()
            result_str = [f'{result:.2f}' for result in results]
            result_str = ', '.join(result_str)
            self.logger.log(f'Total step: {total_steps}, time: {elapsed:.2f} s, '
                            f'max_reward: ' +
                            prColor(f'{max_reward:.3f}', fore=prValuedColor(max_reward, env_min, env_max, 40,
                                                                            "#600000", "#00F0F0")) +
                            f', results: {result_str}')

            prev_actor_steps = current_actor_step

            if eval_steps >= 50000:
                eval_t, eval_reward = self.run()
                self.summary.add_scalar('test/mean', eval_reward, total_steps)
                if eval_reward > eval_max_reward:
                    eval_max_reward = eval_reward
                    self.summary.add_scalar('test/max', eval_reward, total_steps)

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
