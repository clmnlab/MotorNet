import os
import sys 
import torch as th
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
from tqdm import tqdm
from torch.distributions import Normal
from utils import load_stuff
from utils import calculate_angles_between_vectors, calculate_lateral_deviation
from concurrent.futures import ProcessPoolExecutor, as_completed
from networks import GRUPolicy, ActorCriticGRU, GRUActor, GRUCritic, GRUSACActor
from buffer import RolloutBuffer, ReplayBufferOffPolicy
# from networks.py와 buffer.py에서 필요한 클래스를 가져옵니다.
# 이 코드는 Soft Actor-Critic (SAC) 에이전트를 구현합니다.
# SAC 에이전트는 연속적인 행동 공간을 가진 강화 학습 문제를 해결하기 위해 설계되었습니다.
# 이 에이전트는 정책 네트워크와 Q-네트워크를 사용하여 최적의 행동을 선택하고 학습합니다.
# 이 코드는 PyTorch를 사용하여 구현되었으며, GRU 기반의 정책 네트워크와 MLP 기반의 Q-네트워크를 포함합니다. 


class SLAgent:
    def __init__(self, obs_dim, action_dim, batch_size = 32, hidden_dims=128, device='cpu', lr=3e-4, 
                 loss_weight=None, gamma=0.99, tau=0.005, alpha=0.2,
                 freeze_output_layer=False, freeze_input_layer=False, learn_h0=True):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau  # target update rate
        self.alpha = alpha  # entropy temperature
        self.network = GRUPolicy(obs_dim, hidden_dims, action_dim, device=device, 
                                       freeze_output_layer=freeze_output_layer, freeze_input_layer=freeze_input_layer).to(device)
        self.policy_optimizer = optim.Adam(self.network.parameters(), lr=lr)
        # self.replay_buffer = ReplayBuffer(obs_dim, action_dim, capacity=1000, device=device)        
        self.loss_weight = loss_weight if loss_weight is not None else np.array([1e+3,1e+5,1e-1,3e-4,1e-5,1e-3,0])
    def select_action(self, obs, h0=None): ## we can consider "deterministic" option
        h = self.network.init_hidden(batch_size=self.batch_size) if h0 is None else h0
        action, h = self.network(obs, h)
        return action, h
    
    def update(self, data):
        loss, _ = self.calc_loss(data, self.loss_weight)
        loss.to(self.device)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.item()
    
    def save(self, save_dir):
        th.save(self.network.state_dict(), f"{save_dir}")
        # print("done.")

    def load(self, load_dir):
        self.network.load_state_dict(th.load(f"{load_dir}"))
    
    def calc_loss(self, data, loss_weight):
        loss = {
            'position': None,
            'jerk': None,
            'muscle': None,
            'muscle_derivative': None,
            'hidden': None,
            'hidden_derivative': None,
            'hidden_jerk': None,}

        loss['position'] = th.mean(th.sum(th.abs(data['xy']-data['tg']), dim=-1),axis=1) # average over time not targets
        loss['jerk'] = th.mean(th.sum(th.square(th.diff(data['vel'], n=2, dim=1)), dim=-1))
        loss['muscle'] = th.mean(th.sum(data['all_force'], dim=-1))
        loss['muscle_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_force'], n=1, dim=1)), dim=-1))
        loss['hidden'] = th.mean(th.square(data['all_hidden']))
        loss['hidden_derivative'] = th.mean(th.square(th.diff(data['all_hidden'], n=1, dim=1)))
        loss['hidden_jerk'] = th.mean(th.square(th.diff(data['all_hidden'], n=3, dim=1)))
        
        # if loss_weight is None:
        #     # currently in use
        #     loss_weight = np.array([1e+3,1e+5,1e-1,3e-4,1e-5,1e-3,0]) # 3.16227766e-04
            
        loss_weighted = {
            'position': loss_weight[0]*loss['position'],
            'jerk': loss_weight[1]*loss['jerk'],
            'muscle': loss_weight[2]*loss['muscle'],
            'muscle_derivative': loss_weight[3]*loss['muscle_derivative'],
            'hidden': loss_weight[4]*loss['hidden'],
            'hidden_derivative': loss_weight[5]*loss['hidden_derivative'],
            'hidden_jerk': loss_weight[6]*loss['hidden_jerk']
        }

        overall_loss = 0
        for key in loss_weighted:
            if key=='position':
                overall_loss += th.mean(loss_weighted[key].to(self.device))
            else:
                overall_loss += loss_weighted[key]

        return overall_loss, loss_weighted

 


class GRUPPOAgent:
    """사용자 정의 PPO 학습 로직을 담은 에이전트 클래스."""
    def __init__(self, env, 
                 hidden_dim=128, 
                 lr=1e-4, 
                 gamma=0.99, 
                 gae_lambda=0.95, 
                 clip_epsilon=0.2, 
                 n_epochs=10, 
                #  batch_size=64, 
                 sequence_length=100,
                 device='cuda'):
        self.env = env
        self.device = th.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        # self.batch_size = batch_size
        self.sequence_length = sequence_length
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.network = ActorCriticGRU(obs_dim, action_dim, hidden_dim, device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
    def select_action(self, obs, hidden_state):
        """환경과 상호작용할 때 행동을 선택합니다."""
        with th.no_grad():
            dist, value, new_hidden_state = self.network(obs, hidden_state)
            action_raw = dist.sample()
            action = th.sigmoid(action_raw)  # Sigmoid를 사용하여 행동을 [0, 1] 범위로 제한합니다.
            log_prob = dist.log_prob(action_raw).sum(axis=-1)
        
        # 환경 및 버퍼와 상호작용하기 위해 numpy/scalar 값으로 변환
        return (action.cpu().numpy(),
                action_raw.cpu().numpy(), # 원본 행동
                value.cpu().numpy(), 
                log_prob.cpu().numpy(), 
                new_hidden_state.detach())
        
    def update(self, buffer: RolloutBuffer):
        """수집된 데이터로 PPO 업데이트를 수행합니다."""
        # 어드밴티지 정규화 (학습 안정성에 중요)
        advantages = th.tensor(buffer.advantages, dtype=th.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # 🔔 [수정] .detach()를 추가하여 계산 그래프에서 분리한 뒤 numpy로 변환
        buffer.advantages = advantages.detach().cpu().numpy()
        
        for _ in range(self.n_epochs):
            # 🔔 [변경점] 버퍼의 이터레이터를 사용하여 미니배치를 간단히 가져옴
            for batch in buffer:
                mb_states = batch['states']
                mb_raw_actions = batch['raw_actions']
                mb_old_log_probs = batch['log_probs']
                mb_advantages = batch['advantages']
                mb_returns = batch['returns']
                mb_h0 = batch['initial_hidden_states']

                h = mb_h0.contiguous()
                new_log_probs_list, values_list, entropy_list = [], [], []
                for t in range(self.sequence_length):
                    # t번째 스텝의 데이터 (mini_batch_size, feature_dim)
                    states_t = mb_states[t]
                    
                    # 이전 스텝의 은닉 상태(h)를 입력으로 사용
                    dist, values_t, h = self.network(states_t, h)

                    # 계산된 결과들을 리스트에 저장
                    new_log_probs_list.append(dist.log_prob(mb_raw_actions[t]).sum(axis=-1))
                    values_list.append(values_t.flatten())
                    entropy_list.append(dist.entropy().mean())
                    
                # 리스트들을 하나의 텐서로 결합
                # (sequence_length, mini_batch_size)
                new_log_probs = th.stack(new_log_probs_list)
                values = th.stack(values_list)
                entropy_loss = -th.stack(entropy_list).mean()

                # --- 손실 계산 ---
                # Value-function loss
                value_loss = nn.MSELoss()(values, mb_returns)
                
                # Policy-gradient loss (PPO-Clip)
                ratio = th.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = th.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -th.min(surr1, surr2).mean()

                # 최종 손실
                total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                # 최적화
                self.optimizer.zero_grad()
                total_loss.backward()
                th.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5) # 그래디언트 클리핑
                self.optimizer.step()
        return policy_loss.item(), value_loss.item(), entropy_loss.item()
    
    def save(self, save_dir):
        th.save(self.network.state_dict(), f"{save_dir}")
        # print("done.")

    def load(self, load_dir):
        self.network.load_state_dict(th.load(f"{load_dir}"))


# ======================================================================================
#  GRU를 사용하는 DDPG 에이전트
# ======================================================================================
class GRUDDPGAgent:
    def __init__(self, obs_dim, action_dim, hidden_dim, device, lr=1e-3, gamma=0.99, tau=0.005):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.actor = GRUActor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_target = GRUActor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = GRUCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic_target = GRUCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, obs, hidden_state):
        """환경과 상호작용할 때 행동과 다음 은닉 상태를 반환합니다."""
        with th.no_grad():
            # obs의 배치 차원이 1이 아닐 수 있으므로 unsqueeze 대신 reshape 사용
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
            action, next_hidden_state = self.actor(obs_tensor, hidden_state)
        return action.cpu().numpy(), next_hidden_state

    def update(self, batch):
        states, actions, rewards, next_states, dones, hidden_states = [torch.tensor(b).to(self.device) for b in batch]
        
        # hidden_states의 차원을 GRU 입력에 맞게 조정 (batch, 1, hidden_dim) -> (1, batch, hidden_dim)
        hidden_states = hidden_states.permute(1, 0, 2).contiguous()

        # --- 크리틱 업데이트 ---
        with th.no_grad():
            # 타겟 액터에 다음 상태와 '현재' 은닉 상태를 넣어 다음 행동 예측
            next_actions, _ = self.actor_target(next_states, hidden_states)
            target_q = self.critic_target(next_states, next_actions, hidden_states)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        current_q = self.critic(states, actions, hidden_states)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- 액터 업데이트 ---
        # 액터가 생성한 행동을 크리틱이 평가
        new_actions, _ = self.actor(states, hidden_states)
        actor_loss = -self.critic(states, new_actions, hidden_states).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 타겟 네트워크 소프트 업데이트 ---
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, target, source):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

# ======================================================================================
# GRU를 사용하는 SAC 에이전트
# ======================================================================================
class GRUSACAgent:
    def __init__(self, obs_dim, action_dim, hidden_dim, device, lr=3e-4, gamma=0.99, tau=0.005):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.actor = GRUSACActor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic1 = GRUCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic1_target = GRUCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)

        self.critic2 = GRUCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.critic2_target = GRUCritic(obs_dim, action_dim, hidden_dim).to(device)
        self.cri