import numpy as np
import torch as th

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rews_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr, self.size = 0, 0
    def add(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = action
        self.rews_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=th.tensor(self.obs_buf[idxs], device=self.device),
            action=th.tensor(self.acts_buf[idxs], device=self.device),
            reward=th.tensor(self.rews_buf[idxs], device=self.device),
            next_obs=th.tensor(self.next_obs_buf[idxs], device=self.device),
            done=th.tensor(self.done_buf[idxs], device=self.device),
        )
        return batch
