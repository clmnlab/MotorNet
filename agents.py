import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from networks import GRUPolicy, QNetwork
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
    def __init__(self, obs_dim, action_dim, hidden_dims=128, device='cpu', lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 freeze_output_layer=False, freeze_input_layer=False, learn_h0=True):
        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau  # target update rate
        self.alpha = alpha  # entropy temperature
        self.policy_net = GRUPolicy(obs_dim, hidden_dims, action_dim, device=device, 
                                       freeze_output_layer=freeze_output_layer, freeze_input_layer=freeze_input_layer).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, capacity=1000, device=device)        

    def select_action(self, obs, h0=None): ## we can consider "deterministic" option
        h = self.policy_net.init_hidden(batch_size=self.batch_size) if h0 is None else h0
        action, h = self.policy_net(obs, h)
        return action, h
    
    def update(self, data):
        loss, _ = self.calc_loss(data)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        return loss
    


    def save(self, save_dir):
        th.save(self.policy_net.state_dict(), f"{save_dir}/policy.pth")
        print("done.")

    def load(self, load_dir):
        self.policy_net.load_state_dict(th.load(f"{load_dir}/policy.pth"))
    
    def calc_loss(self, data, loss_weight=None):
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
        
        if loss_weight is None:
            # currently in use
            loss_weight = np.array([1e+3,1e+5,1e-1,3e-4,1e-5,1e-3,0]) # 3.16227766e-04
            
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
                overall_loss += th.mean(loss_weighted[key])
            else:
                overall_loss += loss_weighted[key]

        return overall_loss, loss_weighted

 




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
        self.policy_net = GRUPolicyNet(obs_dim, hidden_dims[0], action_dim).to(device)
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
