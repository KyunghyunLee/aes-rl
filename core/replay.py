import ray
import random
import numpy as np
from collections import deque, namedtuple


@ray.remote(resources={'head': 0.01})
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = int(max_size)
        max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, not_done):
        size = state.shape[0]
        flag = False
        remain_size = 0
        if self.ptr + size >= self.max_size:
            new_size = self.max_size - self.ptr
            remain_size = size - new_size
            size = new_size
            flag = True

        self.state[self.ptr:self.ptr+size] = state[0:size]
        self.action[self.ptr:self.ptr + size] = action[0:size]
        self.next_state[self.ptr:self.ptr + size] = next_state[0:size]
        self.reward[self.ptr:self.ptr + size] = reward[0:size]
        self.not_done[self.ptr:self.ptr + size] = not_done[0:size]

        self.ptr = (self.ptr + size) % self.max_size
        self.size = min(self.size + size, self.max_size)

        if flag:
            before_size = size
            size = remain_size
            self.state[self.ptr:self.ptr + size] = state[before_size:]
            self.action[self.ptr:self.ptr + size] = action[before_size:]
            self.next_state[self.ptr:self.ptr + size] = next_state[before_size:]
            self.reward[self.ptr:self.ptr + size] = reward[before_size:]
            self.not_done[self.ptr:self.ptr + size] = not_done[before_size:]

            self.ptr = (self.ptr + size) % self.max_size
            self.size = min(self.size + size, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return self.state[ind], self.action[ind], self.next_state[ind], self.reward[ind], self.not_done[ind]

