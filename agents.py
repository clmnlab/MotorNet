import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from networks import GRUPolicy, ActorCriticGRU, QNetwork
from buffer import ReplayBuffer 
# from networks.py와 buffer.py에서 필요한 클래스를 가져옵니다.
# 이 코드는 Soft Actor-Critic (SAC) 에이전트를 구현합니다.
# SAC 에이전트는 연속적인 행동 공간을 가진 강화 학습 문제를 해결하기 위해 설계되었습니다.
# 이 에이전트는 정책 네트워크와 Q-네트워크를 사용하여 최적의 행동을 선택하고 학습합니다.
# 이 코드는 PyTorch를 사용하여 구현되었으며, GRU 기반의 정책 네트워크와 MLP 기반의 Q-네트워크를 포함합니다. 

import os
import sys 
from utils import load_stuff
from utils import calculate_angles_between_vectors, calculate_lateral_deviation
import torch as th
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


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
        self.policy_net = GRUPolicy(obs_dim, hidden_dims, action_dim, device=device, 
                                       freeze_output_layer=freeze_output_layer, freeze_input_layer=freeze_input_layer).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, capacity=1000, device=device)        
        self.loss_weight = loss_weight if loss_weight is not None else np.array([1e+3,1e+5,1e-1,3e-4,1e-5,1e-3,0])
    def select_action(self, obs, h0=None): ## we can consider "deterministic" option
        h = self.policy_net.init_hidden(batch_size=self.batch_size) if h0 is None else h0
        action, h = self.policy_net(obs, h)
        return action, h
    
    def update(self, data):
        loss, _ = self.calc_loss(data, self.loss_weight)
        loss.to(self.device)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss.item()
    
    def save(self, save_dir):
        th.save(self.policy_net.state_dict(), f"{save_dir}")
        print("done.")

    def load(self, load_dir):
        self.policy_net.load_state_dict(th.load(f"{load_dir}"))
    
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
    def __init__(self, env, hidden_dim=128, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, n_epochs=10, batch_size=64, device='cpu'):
        self.env = env
        self.device = th.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.network = ActorCriticGRU(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
    def select_action(self, obs, hidden_state):
        """환경과 상호작용할 때 행동을 선택합니다."""
        with th.no_grad():
            dist, value, new_hidden_state = self.network(obs, hidden_state)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
        
        # 환경 및 버퍼와 상호작용하기 위해 numpy/scalar 값으로 변환
        return (action.cpu().numpy(), 
                value.cpu().numpy(), 
                log_prob.cpu().numpy(), 
                new_hidden_state.detach())
        
    def update(self, buffer):
        """수집된 데이터로 PPO 업데이트를 수행합니다."""
        # 이점 정규화
        advantages = th.tensor(buffer.advantages, dtype=th.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 데이터를 텐서로 변환
        states = th.tensor(buffer.states, dtype=th.float32).to(self.device)
        actions = th.tensor(buffer.actions, dtype=th.float32).to(self.device)
        old_log_probs = th.tensor(buffer.log_probs, dtype=th.float32).to(self.device)
        returns = th.tensor(buffer.returns, dtype=th.float32).to(self.device)

        for _ in range(self.n_epochs):
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                # 미니배치 생성
                mb_states, mb_actions, mb_old_log_probs, mb_advantages, mb_returns = \
                    states[start:end], actions[start:end], old_log_probs[start:end], advantages[start:end], returns[start:end]

                h0 = self.network.init_hidden(mb_states.shape[0], self.device)
                
                dist, values, _ = self.network(mb_states, h0.squeeze(0))
                
                # 손실 계산
                value_loss = nn.MSELoss()(values.squeeze(), mb_returns)
                
                new_log_probs = dist.log_prob(mb_actions).sum(axis=-1)
                ratio = th.exp(new_log_probs - mb_old_log_probs)
                
                surr1 = ratio * mb_advantages
                surr2 = th.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                policy_loss = -th.min(surr1, surr2).mean()
                
                entropy_loss = -dist.entropy().mean()
                
                total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                # 최적화
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()




class SACAgent:
    def __init__(self, obs_dim, action_dim, hidden_dims, device,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau  # target update rate
        self.alpha = alpha  # entropy temperature

        # Networks
        self.policy_net = GRUPolicy(obs_dim, hidden_dims, action_dim, device=device, 
                                       freeze_output_layer=freeze_output_layer, freeze_input_layer=freeze_input_layer).to(device)
        self.q1_net = QNetwork(obs_dim, action_dim, hidden_dims).to(device)
        self.q2_net = QNetwork(obs_dim, action_dim, hidden_dims).to(device)
        self.q1_target = QNetwork(obs_dim, action_dim, hidden_dims).to(device)
        self.q2_target = QNetwork(obs_dim, action_dim, hidden_dims).to(device)
        # copy weights
        self.q1_target.load_state_dict(self.q1_net.state_dict())
        self.q2_target.load_state_dict(self.q2_net.state_dict())

        # Optimizers
        self.pi_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1_net.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2_net.parameters(), lr=lr)
        # Optional: alpha 자동 조정 기능 등 추가 가능

    def select_action(self, obs, h0=None, deterministic=False):
        # obs: torch.Tensor (obs_dim,) or (batch, obs_dim)
        obs_t = obs.unsqueeze(0) if obs.ndim == 1 else obs
        mean, log_std, h1 = self.policy_net(obs_t, h0)
        std = th.exp(log_std)
        if deterministic:
            action = mean
            log_prob = None
        else:
            dist = Normal(mean, std)
            x = dist.rsample()
            log_prob = dist.log_prob(x).sum(dim=-1, keepdim=True)
            action = x
        # 필요한 범위로 clamp 혹은 tanh 후 log_prob 보정
        return action.squeeze(0), log_prob, h1

    def update(self, replay_buffer, batch_size):
        # 샘플 배치
        batch = replay_buffer.sample_batch(batch_size)
        obs = batch['obs']       # (B, obs_dim)
        action = batch['action'] # (B, action_dim)
        reward = batch['reward'] # (B,1)
        next_obs = batch['next_obs']
        done = batch['done']     # (B,1)

        # 1) Q-network 업데이트
        with th.no_grad():
            # 다음 행동 샘플링
            next_mean, next_log_std, _ = self.policy_net(next_obs)
            next_std = th.exp(next_log_std)
            next_dist = Normal(next_mean, next_std)
            next_action = next_dist.rsample()
            next_log_prob = next_dist.log_prob(next_action).sum(dim=-1, keepdim=True)
            # 타깃 Q 값
            q1_targ = self.q1_target(next_obs, next_action)
            q2_targ = self.q2_target(next_obs, next_action)
            q_targ_min = th.min(q1_targ, q2_targ)
            target_Q = reward + self.gamma * (1 - done) * (q_targ_min - self.alpha * next_log_prob)
        # 현재 Q 예측
        q1_pred = self.q1_net(obs, action)
        q2_pred = self.q2_net(obs, action)
        q1_loss = nn.MSELoss()(q1_pred, target_Q)
        q2_loss = nn.MSELoss()(q2_pred, target_Q)
        self.q1_optimizer.zero_grad(); q1_loss.backward(); self.q1_optimizer.step()
        self.q2_optimizer.zero_grad(); q2_loss.backward(); self.q2_optimizer.step()

        # 2) Policy 업데이트
        mean, log_std, _ = self.policy_net(obs)
        std = th.exp(log_std)
        dist = Normal(mean, std)
        new_action = dist.rsample()
        log_prob = dist.log_prob(new_action).sum(dim=-1, keepdim=True)
        q1_new = self.q1_net(obs, new_action)
        q2_new = self.q2_net(obs, new_action)
        q_new_min = th.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_prob - q_new_min).mean()
        self.pi_optimizer.zero_grad(); policy_loss.backward(); self.pi_optimizer.step()

        # 3) Target network soft update
        for param, target_param in zip(self.q1_net.parameters(), self.q1_target.parameters()):
            target_param.data.copy_( self.tau * param.data + (1 - self.tau) * target_param.data )
        for param, target_param in zip(self.q2_net.parameters(), self.q2_target.parameters()):
            target_param.data.copy_( self.tau * param.data + (1 - self.tau) * target_param.data )

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item()
        }

    def save(self, save_dir):
        th.save(self.policy_net.state_dict(), f"{save_dir}/policy.pth")
        th.save(self.q1_net.state_dict(), f"{save_dir}/q1.pth")
        th.save(self.q2_net.state_dict(), f"{save_dir}/q2.pth")
        # 타깃 네트워크는 학습 시 매번 덮어쓰므로 별도 저장 필요 없음

    def load(self, load_dir):
        self.policy_net.load_state_dict(th.load(f"{load_dir}/policy.pth"))
        self.q1_net.load_state_dict(th.load(f"{load_dir}/q1.pth"))
        self.q2_net.load_state_dict(th.load(f"{load_dir}/q2.pth"))
        # 타깃 네트워크 초기화
        self.q1_target.load_state_dict(self.q1_net.state_dict())
        self.q2_target.load_state_dict(self.q2_net.state_dict())
