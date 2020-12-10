import argparse
import json

import torch.nn as nn
import gym

from .color_console import prAuto
from .argument import *


def init_argument_parser():
    parser = argparse.ArgumentParser(description='pytorch implementation for AES-RL',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env_name', default='HalfCheetah-v2', help='MuJoCo environment name')
    parser.add_argument('--debug', default=False, action='store_true')
    # Basic settings
    parser.add_argument('--logdir', default='./result', help='Directory for logging. ')
    parser.add_argument('--ray_address', help='Ray IP address. ex) 123.123.123.123')
    parser.add_argument('--ray_port', help='Ray port. ex) 12345')
    parser.add_argument('--redis_password', help='Redis-server password')

    parser.add_argument('--max_timestep', default=1e6, type=int, help='Maximum timestep for training')
    parser.add_argument('--seed', default=-1, help='Seed for random. "-1" means not setting a specific seed')

    parser.add_argument('--visualize', action='store_true', help='Pygame visualizer during training')
    parser.add_argument('--config', default='', help='File path for reading arguments when given.')

    # Network architecture parameters
    parser.add_argument('--actor_network_scheme', default='TD3', choices=['TD3'],
                        help='Actor network scheme. Currently, only TD3 is available')
    parser.add_argument('--actor_activation', default='tanh', choices=['relu', 'tanh', 'leaky_relu'],
                        help='Activation layer for actor network')
    parser.add_argument('--critic_activation', default='leaky_relu', choices=['relu', 'tanh', 'leaky_relu'],
                        help='Activation layer for critic network')
    parser.add_argument('--hidden', default=[400, 300], nargs='+', type=int,
                        help='Hidden layer number from top to bottom')

    # Common RL parameters
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=100, type=int, help='Batch size')
    parser.add_argument('--expl_noise', default=0.1, type=float, help='Exploration noise')
    parser.add_argument('--policy_noise', default=0.2, type=float, help='Policy noise')
    parser.add_argument('--noise_clip', default=0.5, type=float, help='Noise clip')
    parser.add_argument('--discount', default=0.99, type=float, help='Discount factor')
    parser.add_argument('--tau', default=0.005, type=float, help='Target network update ratio')
    parser.add_argument('--update_freq', default=2, type=int, help='Actor network update freq')

    # Common Algorithm parameters
    parser.add_argument('--algorithm', default='AES-RL',
                        choices=['TD3', 'CEM-RL', 'P-CEM-RL', 'AES-RL', '(1+1)-ES', 'ACEM-RL'],
                        help='Learning algorithm')
    parser.add_argument('--num_actor_worker', default=1, type=int, help='Number of parallel actor workers')
    parser.add_argument('--num_critic_worker', default=0, type=int, help='Number of parallel critic worker')
    parser.add_argument('--parallel_critic', action='store_true',
                        help='Only for P-CEM-RL, whether to use parallel critic. '
                             'For AES-RL, it always use parallel critic.')
    parser.add_argument('--population_size', default=0, type=int, help='Only for CEM-RL, P-CEM-RL and ACEM-RL. '
                                                                       'Population size')
    parser.add_argument('--replay_size', default=200000, type=int, help='Replay buffer size')
    parser.add_argument('--initial_steps', default=10000, type=int, help='Initial timesteps')

    parser.add_argument('--sigma_init', default=0.001, type=float, help='Initial variance')

    # Algorithm specific parameters
    parser.add_argument('--aesrl_mean_update', default='baseline-relative',
                        choices=['fixed-linear', 'fixed-sigmoid', 'baseline-absolute', 'baseline-relative'],
                        help='Mean update scheme for AES-RL')
    parser.add_argument('--aesrl_mean_update_param', default=0, type=float, help='Mean update parameter for AES-RL')
    parser.add_argument('--aesrl_var_update', default='adaptive',
                        choices=['fixed', 'adaptive'])

    parser.add_argument('--aesrl_rl_ratio', default=0.0, type=float, help='Only for AES-RL. Ratio of RL workers')
    parser.add_argument('--aesrl_rl_k', default=50.0, type=float, help='K value for population control')

    parser.add_argument('--aesrl_negative_ratio', default=0.0, type=float, help='Ratio of negative move')
    parser.add_argument('--aesrl_fixed_var_n', default=0, type=int, help='n for fixed variance update')

    parser.add_argument('--aesrl_one_plus_one_success_rate', default=0.2, type=float)
    parser.add_argument('--aesrl_one_plus_one_success_rate_tau', default=0.99, type=float)
    parser.add_argument('--aesrl_sigma_limit', default=1e-5, type=float)
    parser.add_argument('--aesrl_critic_batch', default=1000, type=int)

    parser.add_argument('--cemrl_damp_init', default=1e-3, type=float, help='Damp init')
    parser.add_argument('--cemrl_damp_limit', default=1e-5, type=float, help='Damp limit')
    parser.add_argument('--cemrl_damp_tau', default=0.95, type=float, help='Damp tau')
    parser.add_argument('--cemrl_tau', default=0.90, type=float, help='CEMRL tau')
    parser.add_argument('--cemrl_no_antithetic', action='store_true', help='CEMRL antithetic')
    parser.add_argument('--cemrl_rl_population', default=0, type=int, help='CEMRL RL population')

    args = parser.parse_args()
    defaults = {}
    for key in vars(args):
        defaults[key] = parser.get_default(key)

    ok_flag = True

    if args.config != '':
        with open(args.config, 'r') as f:
            args_dict = json.load(f)

        for key in args_dict:
            if args.__getattribute__(key) == defaults[key]:
                args.__setattr__(key, args_dict[key])
            else:
                continue

    return args, ok_flag


def init_argument_checker():
    argcheck = ArgCheck()
    argcheck.add_condition(ArgCondition(ArgStatement('algorithm', 'in', ['TD3', 'CEM-RL']),
                                        ArgStatement('num_actor_worker', 'equal', 1)))
    argcheck.add_condition(ArgCondition(ArgStatement('num_actor_worker', 'greater equal', 1)))
    argcheck.add_condition(ArgCondition(ArgStatement('algorithm', 'in', ['CEM-RL', 'P-CEM-RL', 'ACEM-RL']),
                                        ArgStatement('population_size', 'greater equal', 2),
                                        ArgStatement('population_size', 'equal', 0)))
    argcheck.add_condition(ArgCondition(ArgStatement('algorithm', 'not equal', 'P-CEM-RL'),
                                        ArgStatement('parallel_critic', 'equal', False)))
    argcheck.add_condition(ArgCondition(ArgStatement('algorithm', 'equal', 'AES-RL'),
                                        ArgStatement('num_critic_worker', 'greater', 0)))
    argcheck.add_condition(ArgCondition(AndStatement([ArgStatement('algorithm', 'equal', 'P-CEM-RL'),
                                                      ArgStatement('parallel_critic', 'equal', False)]),
                                        ArgStatement('num_critic_worker', 'equal', 0)))
    argcheck.add_condition(ArgCondition(AndStatement([ArgStatement('algorithm', 'equal', 'P-CEM-RL'),
                                                      ArgStatement('parallel_critic', 'equal', True)]),
                                        ArgStatement('num_critic_worker', 'greater equal', 1)))
    argcheck.add_condition(ArgCondition(ArgStatement('algorithm', 'not in',
                                                     ['AES-RL', '(1+1)-ES', 'P-CEM-RL', 'ACEM-RL']),
                                        ArgStatement('num_critic_worker', 'equal', 0)))
    argcheck.add_condition(ArgCondition(ArgStatement('algorithm', 'in', ['AES-RL', '(1+1)-ES', 'ACEM-RL']),
                                        ArgStatement('aesrl_rl_ratio', 'greater', 0.0),
                                        ArgStatement('aesrl_rl_ratio', 'equal', 0.0)))
    argcheck.add_condition(ArgCondition(ArgStatement('ray_address', 'not equal', None),
                                        ArgStatement('ray_port', 'not equal', None)))  # todo: confirm
    argcheck.add_condition(ArgCondition(ArgStatement('algorithm', 'in', ['CEM-RL', 'P-CEM-RL']),
                                        ArgStatement('cemrl_rl_population', 'greater', 0)))
    argcheck.add_condition(ArgCondition(ArgStatement('aesrl_var_update', 'equal', 'fixed'),
                                        ArgStatement('aesrl_fixed_var_n', 'greater', 0)))

    return argcheck


def get_activation_from_string(activation):
    if activation == 'relu':
        return nn.ReLU()
    if activation == 'tanh':
        return nn.Tanh()
    if activation == 'leaky_relu':
        return nn.LeakyReLU()
    raise NotImplementedError


def init_gym_from_args(args):
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    return env, state_dim, action_dim


def get_env_min_max(env_name):
    """
    Used only for color consoles
    :param env_name:
    :return:
    """
    env_min_max = {
        'HalfCheetah-v2': [-2000, 13000],
        'Hopper-v2': [0, 4000],
        'Walker2d-v2': [0, 6500],
        'Ant-v2': [0, 6000],
        'Humanoid-v2': [0, 7000],
        'Swimmer-v2': [0,350],
    }
    if env_name in env_min_max:
        return env_min_max[env_name]
    return 0, 1000
