from utils.util import init_gym_from_args
from utils.logger import Printer


class Algorithm:
    def __init__(self, args):
        self.args = args
        self.replay = None
        self.summary = None
        self.logger = Printer()

        self.critic_workers = []
        self.actor_workers = []
        self.population = None

        self.critic_worker_num = 0
        self.actor_worker_num = 0
        self.individual_dim = 0

        env, self.state_dim, self.action_dim = init_gym_from_args(args)
        env.close()

    def set_logger(self, logger, summary):
        self.logger = logger
        self.summary = summary

    def set_replay(self, replay):
        self.replay = replay

    def get_workers(self):
        return self.critic_worker_num, self.actor_worker_num

    def learn(self):
        raise NotImplementedError

    def get_log_folder_name(self):
        return ''
