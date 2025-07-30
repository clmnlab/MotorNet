import numpy as np
import torch as th

class RolloutBuffer:
    """
    미니배치 샘플링 기능이 포함된 롤아웃 버퍼.
    __iter__ 메서드를 구현하여 for 루프에서 직접 미니배치를 얻을 수 있습니다.
    """
    def __init__(self,
                # 환경에 따라 필수적으로 지정해야 하는 파라미터
                obs_dim: int,
                action_dim: int,
                # 일반적인 기본값이 있는 하이퍼파라미터
                n_steps: int = 1000,
                batch_size: int = 64,
                gamma: float = 0.99,
                gae_lambda: float = 0.95,
                mini_batch_size: int = 64,
                device: th.device = th.device("cuda" if th.cuda.is_available() else "cpu")):
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.reset()

    def reset(self):
        self.states = np.zeros((self.n_steps, self.batch_size, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.batch_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.ptr = 0

    def add(self, state, action, reward, done, log_prob, value):
        if self.ptr >= self.n_steps:
            raise ValueError("RolloutBuffer is full.")
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value.flatten()
        self.ptr += 1

    def compute_returns_and_advantages(self, last_values, last_dones):
        last_values = last_values.detach().cpu().numpy().flatten()
        last_gae_lam = 0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns = self.advantages + self.values

    def __iter__(self):
        """버퍼를 반복 가능하게 만들어 미니배치를 생성합니다."""
        # 1. 데이터를 (n_steps * batch_size, dim) 형태로 평탄화
        total_size = self.n_steps * self.batch_size
        states = self.states.reshape(total_size, self.obs_dim)
        actions = self.actions.reshape(total_size, self.action_dim)
        log_probs = self.log_probs.reshape(total_size)
        advantages = self.advantages.reshape(total_size)
        returns = self.returns.reshape(total_size)
        values = self.values.reshape(total_size)

        # 2. 데이터 인덱스를 무작위로 섞기
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        # 3. 섞인 인덱스를 사용해 미니배치 생성 및 반환(yield)
        for start in range(0, total_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mini_batch_indices = indices[start:end]

            yield {
                'states': th.tensor(states[mini_batch_indices], dtype=th.float32).to(self.device),
                'actions': th.tensor(actions[mini_batch_indices], dtype=th.float32).to(self.device),
                'log_probs': th.tensor(log_probs[mini_batch_indices], dtype=th.float32).to(self.device),
                'advantages': th.tensor(advantages[mini_batch_indices], dtype=th.float32).to(self.device),
                'returns': th.tensor(returns[mini_batch_indices], dtype=th.float32).to(self.device),
                'values': th.tensor(values[mini_batch_indices], dtype=th.float32).to(self.device)
            }