import ray

import numpy as np
import datetime

from core.population import RLPopulation
from .common import Algorithm
from utils.color_console import prColor, prValuedColor
from utils.util import get_env_min_max


class Td3Algorithm(Algorithm):
    def __init__(self, args):
        super(Td3Algorithm, self).__init__(args)

        self.max_timestep = args.max_timestep
        self.initial_steps = args.initial_steps

        self.critic_worker_num = 0
        self.actor_worker_num = 1

    def get_log_folder_name(self):
        return f'TD3'

    def assign_workers(self, critic_workers, actor_workers):
        self.critic_workers = critic_workers
        self.actor_workers = actor_workers

        self.individual_dim = ray.get(self.actor_workers[0].count_parameters.remote(name='main'))
        mean = ray.get(self.actor_workers[0].get_weight.remote(name='main'))
        self.population = RLPopulation(self.individual_dim, mean, self.args)

    def learn(self):
        total_steps = 0
        episode_count = 0
        ray.get([worker.set_train.remote() for worker in self.critic_workers + self.actor_workers])
        init_param = self.population.ask(1)

        ray.get([worker.set_weight.remote(init_param, name='main')
                 for worker in self.critic_workers + self.actor_workers])
        ray.get([worker.set_weight.remote(init_param, name='target')
                 for worker in self.critic_workers + self.actor_workers])

        max_reward = -10000
        eval_max_reward = -10000
        init_time = datetime.datetime.now()
        eval_step = 0

        env_min, env_max = get_env_min_max(self.args.env_name)

        while total_steps < self.max_timestep:
            if total_steps < self.initial_steps:
                rollout_obj = self.actor_workers[0].rollout.remote(random_action=True, eval=False, mid_train=False)
            else:
                rollout_obj = self.actor_workers[0].rollout.remote(random_action=False, eval=False, mid_train=True,
                                                                   noise=True, learn_critic=True, learn_actor=True)
                # todo: confirm.
            episode_t, episode_reward = ray.get(rollout_obj)
            episode_count += 1
            total_steps += episode_t
            if episode_reward > max_reward:
                max_reward = episode_reward
                self.summary.add_scalar('train/max', episode_reward, total_steps)
            if episode_count % 1 == 0:
                # todo: change to reporter
                elapsed = (datetime.datetime.now() - init_time).total_seconds()
                self.logger.log(f'Total step: {total_steps}, time: {elapsed:.2f} s, '
                                f'max_reward: ' +
                                prColor(f'{max_reward:.3f}', fore=prValuedColor(max_reward, env_min, env_max, 40,
                                                                                "#600000", "#00F0F0")) +
                                f', current_reward: ' +
                                prColor(f'{episode_reward:.3f}', fore=prValuedColor(episode_reward, env_min, env_max,
                                                                                    40, "#600000", "#00F0F0")))

            self.summary.add_scalar('train/individuals', episode_reward, total_steps)

            eval_step += episode_t

            if eval_step >= 100000:
                eval_t, eval_reward = self.run()
                if eval_reward > eval_max_reward:
                    eval_max_reward = eval_reward
                    self.summary.add_scalar('test/max', eval_reward, total_steps)
                self.logger.log(f'Evaluation: {eval_reward}')
                self.summary.add_scalar('test/mean', eval_reward, total_steps)
                eval_step = 0

    def run(self):
        ray.get([worker.set_eval.remote() for worker in self.critic_workers + self.actor_workers])
        result = ray.get(self.actor_workers[0].rollout.remote(random_action=False, eval=True, mid_train=False))
        ray.get([worker.set_train.remote() for worker in self.critic_workers + self.actor_workers])
        return result

