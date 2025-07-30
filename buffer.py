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
        self.raw_actions = np.zeros((self.n_steps, self.batch_size, self.action_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.batch_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.ptr = 0

    def add(self, state, raw_action, action, reward, done, log_prob, value):
        if self.ptr >= self.n_steps:
            raise ValueError("RolloutBuffer is full.")
        self.states[self.ptr] = state
        self.raw_actions[self.ptr] = raw_action
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
    
    # def get_batch(self):
    #     """
    #     수집된 데이터를 하나의 큰 배치로 만들고 PyTorch 텐서로 변환합니다.
    #     """
    #     # (n_steps, batch_size, *shape) -> (n_steps * batch_size, *shape)
    #     states = self.states.reshape(-1, self.obs_dim)
    #     actions = self.actions.reshape(-1, self.action_dim)
    #     log_probs = self.log_probs.reshape(-1)
    #     advantages = self.advantages.reshape(-1)
    #     returns = self.returns.reshape(-1)
    #     values = self.values.reshape(-1)

    #     # PyTorch 텐서로 변환
    #     states = th.tensor(states, dtype=th.float32).to(self.device)
    #     actions = th.tensor(actions, dtype=th.float32).to(self.device)
    #     log_probs = th.tensor(log_probs, dtype=th.float32).to(self.device)
    #     advantages = th.tensor(advantages, dtype=th.float32).to(self.device)
    #     returns = th.tensor(returns, dtype=th.float32).to(self.device)
    #     values = th.tensor(values, dtype=th.float32).to(self.device)

    #     return {
    #         'states': states,
    #         'actions': actions,
    #         'log_probs': log_probs,
    #         'advantages': advantages,
    #         'returns': returns,
    #         'values': values
    #     }
        
    def __iter__(self):
        """버퍼를 반복 가능하게 만들어 미니배치를 생성합니다."""
        total_size = self.n_steps * self.batch_size
        
        # 데이터를 (total_size, dim) 형태로 평탄화
        states = self.states.reshape(total_size, self.obs_dim)
        # 🔔 [변경점] raw_actions도 평탄화
        raw_actions = self.raw_actions.reshape(total_size, self.action_dim)
        log_probs = self.log_probs.reshape(total_size)
        advantages = self.advantages.reshape(total_size)
        returns = self.returns.reshape(total_size)

        indices = np.arange(total_size)
        np.random.shuffle(indices)

        for start in range(0, total_size, self.mini_batch_size):
            end = start + self.mini_batch_size
            mini_batch_indices = indices[start:end]

            # 🔔 [변경점] 반환하는 딕셔너리에 'raw_actions' 추가
            yield {
                'states': th.tensor(states[mini_batch_indices], dtype=th.float32).to(self.device),
                'raw_actions': th.tensor(raw_actions[mini_batch_indices], dtype=th.float32).to(self.device),
                'log_probs': th.tensor(log_probs[mini_batch_indices], dtype=th.float32).to(self.device),
                'advantages': th.tensor(advantages[mini_batch_indices], dtype=th.float32).to(self.device),
                'returns': th.tensor(returns[mini_batch_indices], dtype=th.float32).to(self.device),
            }
            
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