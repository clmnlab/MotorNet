import numpy as np
import torch
import torch.nn as nn

# ======================================================================================
# 1. 테스트 대상 버퍼 코드 (Canvas에 있는 코드를 그대로 가져옴)
# ======================================================================================
class RolloutBuffer:
    """
    순환 신경망(RNN/GRU) 학습을 위한 롤아웃 버퍼 (배치 데이터 지원).
    데이터의 시간적 순서를 유지하고, 학습 시 연속된 시퀀스 단위의 미니배치를 생성합니다.
    """
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 batch_size: int,              # 롤아웃 시 한 스텝에 처리하는 데이터의 배치 크기
                 sequence_length: int = 100,
                 n_steps: int = 100,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 mini_batch_size: int = 16,    # [의미 변경] 한 미니배치에 포함될 '시퀀스의 개수'
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.reset()

    def reset(self):
        self.states = np.zeros((self.n_steps, self.batch_size, self.obs_dim), dtype=np.float32)
        # CartPole은 이산 행동 공간이므로 action은 스칼라, raw_action은 로짓(2)
        self.raw_actions = np.zeros((self.n_steps, self.batch_size, self.action_dim), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.batch_size, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.dones = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.log_probs = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.values = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.returns = np.zeros((self.n_steps, self.batch_size), dtype=np.float32)
        self.hidden_states = np.zeros((self.n_steps, self.batch_size, self.hidden_dim), dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, state, raw_action, action, reward, done, log_prob, value, hidden_state):
        if self.ptr >= self.n_steps:
            self.full = True
            return

        self.states[self.ptr] = state
        self.raw_actions[self.ptr] = raw_action
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value.flatten()
        self.hidden_states[self.ptr] = hidden_state.detach().cpu().numpy().reshape(self.batch_size, self.hidden_dim)
        self.ptr = (self.ptr + 1)

    def compute_returns_and_advantages(self, last_values, last_dones):
        """GAE 계산. last_values/dones는 (batch_size,) 형태의 벡터입니다."""
        # 🔔 [수정] 입력값이 스칼라일 경우를 대비하여 배열로 변환하는 로직 추가
        last_values = np.asarray(last_values).flatten()
        last_dones = np.asarray(last_dones).flatten()      
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
        all_data = {
            'states': torch.tensor(self.states, dtype=torch.float32).swapaxes(0, 1),
            'raw_actions': torch.tensor(self.raw_actions, dtype=torch.float32).swapaxes(0, 1),
            'actions': torch.tensor(self.actions, dtype=torch.float32).swapaxes(0, 1),
            'log_probs': torch.tensor(self.log_probs, dtype=torch.float32).swapaxes(0, 1),
            'advantages': torch.tensor(self.advantages, dtype=torch.float32).swapaxes(0, 1),
            'returns': torch.tensor(self.returns, dtype=torch.float32).swapaxes(0, 1),
            'hidden_states': torch.tensor(self.hidden_states, dtype=torch.float32).swapaxes(0, 1),
        }

        n_sequences_per_stream = self.n_steps // self.sequence_length
        num_total_sequences = self.batch_size * n_sequences_per_stream
        sequences_per_batch = self.mini_batch_size

        indices = np.arange(num_total_sequences)
        np.random.shuffle(indices)

        for i in range(0, num_total_sequences, sequences_per_batch):
            batch_indices = indices[i : i + sequences_per_batch]
            batch = {}
            for key, tensor in all_data.items():
                sequences = tensor.reshape(num_total_sequences, self.sequence_length, *tensor.shape[2:])
                batch[key] = sequences[batch_indices].to(self.device)

            initial_h = batch['hidden_states'][:, 0, :]
            batch['initial_hidden_states'] = initial_h.unsqueeze(0)

            yield {
                key: tensor.transpose(0, 1) if key != 'initial_hidden_states' else tensor
                for key, tensor in batch.items()
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