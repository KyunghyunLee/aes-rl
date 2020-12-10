import torch
import torch.nn as nn
import torch.nn.functional as F
import ray

import numpy as np
import itertools

from utils.util import get_activation_from_string
from utils.timestamp import Timestamp


class RayModule(nn.Module):
    def __init__(self, device, name=None):
        super(RayModule, self).__init__()
        self.name = name
        self.device = device
        self.layers = []
        self.forward_layer = None
        self.critic_params = None
        self.actor_params = None
        self.last = False
        self.max_action = 1.0

        self.actor_names = []
        self.critic_names = []

    def set_max_action(self, max_action):
        self.max_action = max_action
        for layer in self.layers:
            if isinstance(layer, RayModule):
                layer.set_max_action(max_action)

    def set_last(self, last):
        self.last = last

    def set_eval(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def set_train(self):
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    def get_weight(self, name=None) -> np.ndarray or None:
        params = self.count_parameters(name)
        if params == 0:
            return None
        param = np.zeros(params, dtype=np.float32)
        pos = 0
        name_list = name if isinstance(name, list) else [name]

        for name in name_list:
            leaves = self.get_leaf(name)
            # todo: confirm
            for leaf in leaves:
                for v in leaf.parameters():
                    size1 = v.numel()
                    if size1 == 0:
                        continue
                    param[pos:pos + size1] = v.view(-1).cpu().detach().numpy()
                    pos += size1
        return param

    def set_weight(self, weight, name=None):
        params = self.count_parameters(name)
        assert params == weight.size
        if isinstance(weight, ray.ObjectRef):
            weight = ray.get(weight)

        param = torch.from_numpy(weight).to(self.device)
        pos = 0

        name_list = name if isinstance(name, list) else [name]

        for name in name_list:
            leaves = self.get_leaf(name)
            # todo: confirm
            for leaf in leaves:
                for v in leaf.parameters():
                    size1 = v.numel()
                    if size1 == 0:
                        continue
                    data = param[pos:pos + size1].view(v.size())
                    v.data.copy_(data.data)
                    pos += size1

    def get_node(self, name=None):
        if name is not None:
            if self.name is not None:
                names = name.split('.')
                if self.name == names[0]:
                    name = '.'.join(names[1:]) if len(names) > 1 else None
                else:
                    return None
        if name is None:
            return self

        for layer in self.layers:
            if isinstance(layer, RayModule):
                result = layer.get_node(name)
                if result is not None:
                    return result
        return None

    def get_leaf(self, name=None):
        if name is not None:
            if self.name is not None:
                names = name.split('.')
                if self.name == names[0]:
                    name = '.'.join(names[1:]) if len(names) > 1 else None
                else:
                    return []
        all_layers = []
        for layer in self.layers:
            if isinstance(layer, RayModule):
                all_layers += layer.get_leaf(name)
            else:
                if name is None:
                    all_layers += [layer]

        return all_layers

    def forward(self, *input):
        if self.last:
            return self.max_action * torch.tanh(self.forward_layer(*input))
        return self.forward_layer(*input)

    def count_parameters(self, name=None):
        name_list = name if isinstance(name, list) else [name]
        params = 0
        for name in name_list:
            leaves = self.get_leaf(name)
            for leaf in leaves:
                for v in leaf.parameters():
                    params += v.numel()
        return params

    def learn(self, replay, *args, **kwargs):
        raise NotImplementedError

    def get_critic_names(self):
        return self.critic_names

    def get_actor_names(self):
        return self.actor_names


class Actor(RayModule):
    def __init__(self, device, state_dim, hidden_dim, action_dim, n_layer, activation, name=None):
        super(Actor, self).__init__(device, name)
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim for _ in range(n_layer - 1)]

        for idx in range(n_layer):
            inputs = state_dim if idx == 0 else hidden_dim[idx - 1]
            outputs = action_dim if idx == n_layer - 1 else hidden_dim[idx]

            self.layers.append(nn.Linear(inputs, outputs))
            if idx != n_layer - 1:
                act_layer = get_activation_from_string(activation)
                self.layers.append(act_layer)

        self.forward_layer = nn.Sequential(*self.layers)


class Critic(RayModule):
    def __init__(self, device, state_dim, hidden_dim, action_dim, n_layer, activation, name=None):
        super(Critic, self).__init__(device, name)
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim for _ in range(n_layer - 1)]

        for idx in range(n_layer):
            inputs = state_dim + action_dim if idx == 0 else hidden_dim[idx - 1]
            outputs = 1 if idx == n_layer - 1 else hidden_dim[idx]

            self.layers.append(nn.Linear(inputs, outputs))
            if idx != n_layer - 1:
                act_layer = get_activation_from_string(activation)
                self.layers.append(act_layer)

        self.forward_layer = nn.Sequential(*self.layers)


class TD3(RayModule):
    def __init__(self, device, state_dim, hidden_dim, action_dim, n_layer, args, name=None):
        super(TD3, self).__init__(device, name)
        self.actor = Actor(device, state_dim, hidden_dim, action_dim, n_layer,
                           activation=args.actor_activation, name='actor')
        self.critic1 = Critic(device, state_dim, hidden_dim, action_dim, n_layer,
                              activation=args.critic_activation, name='critic1')
        self.critic2 = Critic(device, state_dim, hidden_dim, action_dim, n_layer,
                              activation=args.critic_activation, name='critic2')
        self.layers = [self.actor, self.critic1, self.critic2]
        self.forward_layer = self.actor

        self.policy_noise = args.policy_noise
        self.expl_noise = args.expl_noise
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.noise_clip = args.noise_clip
        self.lr = args.lr
        self.num_critic_worker = args.num_critic_worker
        self.policy_freq = args.update_freq

        self.critic_optimizer = None
        self.actor_optimizer = None
        self.update_count = 0
        # self.timestamp = Timestamp('./')
        self.critic_names = ['critic1', 'critic2']
        self.actor_names = ['actor']

    def set_last(self, last):
        self.actor.set_last(last)

    def learn(self, replay, *args, **kwargs):
        target_network = kwargs['target_network']
        learn_critic = kwargs['learn_critic']
        learn_actor = kwargs['learn_actor']
        batches = kwargs['batches']

        training_reporter = kwargs['training_reporter']
        critic_optimizer = kwargs['critic_optimizer']
        actor_optimizer = kwargs['actor_optimizer']

        target_critic1 = target_network.get_node(name='target.critic1')
        target_critic2 = target_network.get_node(name='target.critic2')
        assert target_critic1 is not None
        assert target_critic2 is not None

        if learn_critic:
            if self.critic_optimizer is not None and not kwargs['reset_optim']:
                critic_optimizer = self.critic_optimizer
            else:
                critic_optimizer = torch.optim.Adam(
                    itertools.chain(self.critic1.parameters(), self.critic2.parameters()), lr=self.lr)
            self.critic_optimizer = critic_optimizer

        if learn_actor:
            if self.actor_optimizer is not None and not kwargs['reset_optim']:
                actor_optimizer = self.actor_optimizer
            else:
                actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
            self.actor_optimizer = actor_optimizer

        if training_reporter is not None:
            # todo: implement training reporter
            pass

        main_critic1 = self.critic1
        main_critic2 = self.critic2

        critic_loss_float = 0.0
        actor_loss_float = 0.0
        for batch in range(batches):
            actor_learned = False
            critic_learned = False
            state, action, next_state, reward, not_done = ray.get(replay.sample.remote(self.batch_size))

            state = torch.from_numpy(state).float().to(self.device)
            action = torch.from_numpy(action).float().to(self.device)
            next_state = torch.from_numpy(next_state).float().to(self.device)
            reward = torch.from_numpy(reward).float().to(self.device)

            not_done = torch.from_numpy(not_done).float().to(self.device)

            if learn_critic:
                with torch.no_grad():
                    noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

                    action_target = target_network(next_state)

                    action_target += noise

                    next_action = action_target.clamp(-self.max_action, self.max_action)
                    critic_next_state = torch.cat([next_state, next_action], dim=-1)
                    target_Q1 = target_critic1(critic_next_state)
                    target_Q2 = target_critic2(critic_next_state)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + not_done * self.discount * target_Q

                critic_state = torch.cat([state, action], dim=-1)

                current_Q1 = main_critic1(critic_state)
                current_Q2 = main_critic2(critic_state)

                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                critic_loss_float = critic_loss.cpu().detach().item()
                critic_learned = True

            if learn_actor:
                if self.update_count % self.policy_freq == 0 or not learn_critic:
                    current_action = self.actor(state)
                    critic_state = torch.cat([state, current_action], dim=-1)
                    actor_loss = -main_critic1(critic_state).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # nn.utils.clip_grad_norm_(actor_params, clip)
                    actor_optimizer.step()
                    actor_loss_float = actor_loss.cpu().detach().item()
                    actor_learned = True

            if (learn_critic and not learn_actor) or actor_learned:
                for param, target_param in zip(self.named_parameters(), target_network.named_parameters()):
                    k, v = param
                    tk, tv = target_param
                    if not critic_learned and k.startswith('critic'):
                        continue
                    if not actor_learned and k.startswith('actor'):
                        continue
                    tv.data.copy_(self.tau * v.data + (1 - self.tau) * tv.data)
            # self.timestamp.toc('update param')
            self.update_count += 1
            # todo: implement reporter


class DDPG(RayModule):
    def __init__(self, device, state_dim, hidden_dim, action_dim, n_layer, args, name=None):
        super(DDPG, self).__init__(device, name)
        self.actor = Actor(device, state_dim, hidden_dim, action_dim, n_layer, args.actor_activation, name='actor')
        self.critic = Critic(device, state_dim, hidden_dim, action_dim, n_layer, args.critic_activation, name='critic')
        self.layers = [self.actor, self.critic]
        self.forward_layer = self.actor
        self.critic_params = self.critic.parameters()
        self.actor_params = self.actor.parameters()

        self.critic_names = ['critic']
        self.actor_names = ['actor']

    def learn(self, replay, *args, **kwargs):
        raise NotImplementedError

    def set_last(self, last):
        self.actor.set_last(last)


class Agent(RayModule):
    def __init__(self, device, state_dim, hidden_dim, action_dim, n_layer, args, name=None):
        super(Agent, self).__init__(device, name)
        if args.actor_network_scheme == 'TD3':
            self.main = TD3(device, state_dim, hidden_dim, action_dim, n_layer,args, name='main')
            self.target = TD3(device, state_dim, hidden_dim, action_dim, n_layer, args, name='target')
            self.target.load_state_dict(self.main.state_dict())
            self.forward_layer = self.main
            self.layers = [self.main, self.target]
        else:
            raise NotImplementedError

    def learn(self, replay, *args, **kwargs):
        kwargs['target_network'] = self.target
        self.main.learn(replay, *args, **kwargs)

    def set_last(self, last):
        self.main.set_last(last)
        self.target.set_last(last)

    def get_critic_names(self):
        critic_names = self.main.critic_names
        critic_names = ['main.' + name for name in critic_names] + ['target.' + name for name in critic_names]
        return critic_names

    def get_actor_names(self):
        actor_names = self.main.actor_names
        actor_names = ['main.' + name for name in actor_names] + ['target.' + name for name in actor_names]
        return actor_names

    def set_train(self):
        self.main.set_train()

    def set_eval(self):
        self.main.set_eval()
