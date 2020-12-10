import ray
import torch
import numpy as np

import copy

from utils.util import init_gym_from_args
from .network import Agent
from utils.timestamp import Timestamp


class Worker:
    def __init__(self, replay, name, args, logger, reporter=None):
        self.replay = replay
        self.name = name
        self.logger = logger
        self.reporter = reporter

        self.env, self.state_dim, self.action_dim = init_gym_from_args(args)
        self.max_action = float(self.env.action_space.high[0])
        if args.seed == -1:
            args.seed = int(np.random.randint(0, 2 ** 32))
            logger.log(f'Seed changed to: {args.seed}')

        self.env.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # rl params

        self.expl_noise = args.expl_noise

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.network = Agent(self.device, self.state_dim, args.hidden, self.action_dim,
                             len(args.hidden) + 1, args).to(self.device)
        self.network.set_last(True)
        self.network.set_max_action(self.max_action)

        self.default_training_kwargs = {
            'learn_critic': False,
            'learn_actor': False,
            'critic_optimizer' : None,
            'actor_optimizer': None,
            'batches': 1,
            'training_reporter': self.reporter,
            'reset_optim': False,
            'individual_id': 0,
        }
        self.individual_id = 0

    def ping(self):
        gpu = ray.get_gpu_ids()
        print(f'[{self.name}]: GPU {gpu[0]}')
        return None

    def get_weight(self, *args, **kwargs):
        return self.network.get_weight(*args, **kwargs)

    def set_weight(self, *args, **kwargs):
        self.network.set_weight(*args, **kwargs)

    def count_parameters(self, *args, **kwargs):
        return self.network.count_parameters(*args, **kwargs)

    def get_default_training_kwargs(self):
        return self.default_training_kwargs

    def set_individual_id(self, id):
        self.individual_id = id

    def rollout(self, random_action=False, eval=False, mid_train=False, noise=False, **train_kwargs):
        # self.timestamp.tic('rollout')
        state, done = self.env.reset(), False
        total_t = 0
        reward_sum = 0
        state_replay = []
        action_replay = []
        next_state_replay = []
        reward_replay = []
        not_done_replay = []

        while not done:
            # self.timestamp.tic('act')
            if random_action:
                action = self.env.action_space.sample()
            else:
                state_t = torch.from_numpy(state).float().to(self.device)
                with torch.no_grad():
                    action = self.network(state_t).cpu().detach().numpy()
                if noise:
                    action += np.random.normal(0, self.max_action * self.expl_noise, size=self.action_dim)
                action = action.clip(-self.max_action, self.max_action)
                del state_t
            # self.timestamp.toc('act')
            total_t += 1
            # self.timestamp.tic('step')
            next_state, reward, done, info = self.env.step(action)
            done_float = float(done) if total_t < self.env._max_episode_steps else 0.0
            # self.timestamp.toc('step')

            not_done = 1.0 - done_float
            reward_sum += reward

            if not eval:
                state_replay.append(np.expand_dims(state, axis=0).astype(np.float32))
                action_replay.append(np.expand_dims(action, axis=0).astype(np.float32))
                next_state_replay.append(np.expand_dims(next_state, axis=0).astype(np.float32))
                reward_replay.append(np.expand_dims(np.array([reward]), axis=0).astype(np.float32))
                not_done_replay.append(np.expand_dims(np.array([not_done]), axis=0).astype(np.float32))

            if mid_train:
                # self.timestamp.tic('train')
                params = copy.deepcopy(self.default_training_kwargs)
                for key in train_kwargs:
                    if key not in params:
                        raise KeyError
                    params[key] = train_kwargs[key]

                self.train(**params)
                # self.timestamp.toc('train')
            state = next_state

        if not eval:
            # state_obj = [ray.put(replay) for replay in state_replay]
            # action_obj = [ray.put(replay) for replay in action_replay]
            # next_state_obj = [ray.put(replay) for replay in next_state_replay]
            # reward_obj = [ray.put(replay) for replay in reward_replay]
            # not_done_obj = [ray.put(replay) for replay in not_done_replay]
            # self.timestamp.tic('replay')
            state_obj = ray.put(np.concatenate(state_replay, axis=0))
            action_obj = ray.put(np.concatenate(action_replay, axis=0))
            next_state_obj = ray.put(np.concatenate(next_state_replay, axis=0))
            reward_obj = ray.put(np.concatenate(reward_replay, axis=0))
            not_done_obj = ray.put(np.concatenate(not_done_replay, axis=0))
            self.replay.add.remote(state_obj, action_obj, next_state_obj, reward_obj, not_done_obj)
            # self.timestamp.toc('replay')

        # self.timestamp.toc('rollout')
        return total_t, reward_sum

    def train(self, **kwargs):
        return self.network.learn(self.replay, **kwargs)

    def set_train(self):
        self.network.set_train()

    def set_eval(self):
        self.network.set_eval()

    def get_critic_names(self):
        return self.network.get_critic_names()

    def get_actor_names(self):
        return self.network.get_actor_names()
