import os
import json
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from collections import deque
import random
import time
from pathlib import Path

from buffer import ReplayBufferOnPolicy, ReplayBufferOffPolicy
from agents import SLAgent, GRUPPOAgent, GRUDDPGAgent
from task import CentreOutFFGym, CentreOutFF



def run_training(env, agent, buffer, config, save_dir, load_path=None):
    
    agent_type = agent.__class__.__name__
    print(f"--- {agent_type} 알고리즘으로 학습을 시작합니다 ---")

    if load_path and Path(load_path).exists():
        agent.load(load_path)
        print(f"저장된 모델 로드 완료: {load_path}")

    # --- 학습 루프 공통 변수 초기화 ---
    start_time = time.time()
    batch_size = config['env_params']['batch_size']
    total_timesteps = config['training_params']['total_timesteps']
    obs, _ = env.reset(options={'batch_size': batch_size})
    hidden_state = agent.network.init_hidden(batch_size) if hasattr(agent.network, 'init_hidden') else agent.actor.init_hidden(batch_size, agent.device)
    
    episode_rewards = []
    current_episode_rewards = np.zeros(batch_size)
    episode_count = 0

    is_on_policy = isinstance(agent, GRUPPOAgent)

    # --- 메인 학습 루프 ---
    for step in range(total_timesteps):
        # 1. 환경과 상호작용
        if is_on_policy:
            action, raw_action, value, log_prob, next_hidden_state = agent.select_action(obs, hidden_state)
        else: # DDPG/SAC
            action, next_hidden_state = agent.select_action(obs, hidden_state)
            noise = np.random.normal(0, config['training_params']['ddpg'].get('exploration_noise', 0.1), size=action.shape)
            action = np.clip(action + noise, 0, 1.0)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # 2. 버퍼에 데이터 저장
        if is_on_policy:
            buffer.add(obs, raw_action, action, reward, terminated, log_prob, value, hidden_state)
        else:
            for i in range(batch_size):
                buffer.add(obs[i], action[i], reward[i], next_obs[i], terminated[i], hidden_state[:, i, :])
        
        obs = next_obs
        hidden_state = next_hidden_state
        current_episode_rewards += reward

        # 3. 학습 실행 (알고리즘에 따라 다름)
        if is_on_policy:
            if (step + 1) % config['training_params']['ppo']['n_steps'] == 0:
                with th.no_grad():
                    _, last_value, _ = agent.network(th.from_numpy(next_obs).float().to(agent.device), hidden_state)
                buffer.compute_returns_and_advantages(last_value.cpu().numpy(), terminated | truncated)
                ploss, vloss, eloss = agent.update(buffer)
                buffer.reset()
                print(f"스텝 {step+1} | 최근 평균 보상: {np.mean(episode_rewards[-10:]):.2f} | Loss: {ploss:.4f}, {vloss:.4f}, {eloss:.4f}")
        else: # Off-Policy
            if len(buffer) > config['training_params']['ddpg']['learning_starts']:
                batch_data = buffer.sample(config['training_params']['ddpg']['update_batch_size'])
                losses = agent.update(batch_data)
                if step % 1000 == 0:
                    print(f"스텝 {step} | Losses: {losses}")

        # 4. 에피소드 종료 처리
        dones = terminated | truncated
        if dones.any():
            for i, done in enumerate(dones):
                if done:
                    episode_rewards.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0
                    episode_count += 1
            init_hidden_func = getattr(agent.network if hasattr(agent, 'network') else agent.actor, 'init_hidden')
            reset_hidden = init_hidden_func(dones.sum(), agent.device)
            hidden_state[:, dones, :] = reset_hidden
        
        # 5. 모델 저장
        if (episode_count > 0) and (episode_count % 10 == 0):
            agent.save(save_dir / f'agent_epi{episode_count}.pth')
            # ... (보상 저장 로직) ...

    print(f"훈련 완료! (소요 시간: {time.time() - start_time:.2f}초)")
    # ... (결과 시각화) ...

# ======================================================================================
# 5. 스크립트 실행 부분
# ======================================================================================
if __name__ == '__main__':
    config_path = 'params.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # env = CentreOutFFGym(**config['env_params']) # 실제 환경
    
    # --- 🔔 실행할 에이전트 선택 ---
    AGENT_TO_RUN = "PPO"
    # AGENT_TO_RUN = "DDPG"
    
    if AGENT_TO_RUN == "PPO":
        agent_params = config['training_params']['ppo']
        agent = GRUPPOAgent(env, **agent_params)
        buffer = ReplayBufferOnPolicy(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device=agent.device,
            **config['env_params'],
            **agent_params
        )
    elif AGENT_TO_RUN == "DDPG":
        agent_params = config['training_params']['ddpg']
        agent = GRUDDPGAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            device="cuda" if th.cuda.is_available() else "cpu",
            **agent_params
        )
        buffer = ReplayBufferOffPolicy(capacity=agent_params['buffer_capacity'])
    else:
        raise ValueError(f"알 수 없는 에이전트 타입: {AGENT_TO_RUN}")

    run_training(
        env=env,
        agent=agent,
        buffer=buffer,
        config=config,
        save_dir=Path.cwd() / 'results' / f"{AGENT_TO_RUN}_agent",
        load_path=None
    )
