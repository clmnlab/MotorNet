import os
import sys
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import motornet as mn
import utils
import agents
from pathlib import Path
from tqdm import tqdm
import argparse
import json    
import task 
from importlib import reload
reload(agents)
reload(utils)
reload(task)
from utils import load_env, load_policy, calc_loss, run_rollout
from task import CentreOutFFGym
from buffer import RolloutBuffer
from agents import SLAgent, GRUPPOAgent
import time

if __name__ == '__main__':
    # --- 하이퍼파라미터 ---
    TOTAL_TIMESTEPS = 50000
    N_STEPS = 2048  # 데이터 수집 스텝 수
    HIDDEN_DIM = 128
    config = 'params.json'
    with open(config, 'r') as f:
        config = json.load(f)

    # --- parameters setting ---
    env_params = config['env_params']
    train_params = config['training_params']

    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())

    # --- 환경 및 에이전트 생성 ---
    env = CentreOutFFGym(effector=effector, **env_params)
    device = "cuda" if th.cuda.is_available() else "cpu"
    agent = GRUPPOAgent(env, hidden_dim=HIDDEN_DIM, device=device)
    buffer = RolloutBuffer(N_STEPS, env.observation_space.shape[0], env.action_space.shape[0], device, agent.gamma, agent.gae_lambda)

    print(f"훈련 시작 (총 {TOTAL_TIMESTEPS} 스텝) on {device}...")
    start_time = time.time()
    
    obs, _ = env.reset(options={'batch_size': train_params['batch_size']})
    hidden_state = agent.network.init_hidden(train_params['batch_size'], device)
    
    episode_rewards = []
    current_episode_reward = 0
    
    for step in range(TOTAL_TIMESTEPS):
        obs_tensor = th.from_numpy(obs).float().to(device)
        action, value, log_prob, next_hidden_state = agent.select_action(obs_tensor, hidden_state)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        buffer.add(obs, action, reward, terminated, log_prob, value)
        current_episode_reward += reward
        
        obs = next_obs
        hidden_state = next_hidden_state
        
        done = terminated or truncated
        if done:
            episode_rewards.append(current_episode_reward)
            if len(episode_rewards) % 10 == 0:
                print(f"스텝 {step}: 최근 10 에피소드 평균 보상: {np.mean(episode_rewards[-10:]):.2f}")
            current_episode_reward = 0
            obs, _ = env.reset()
            hidden_state = agent.network.init_hidden(1, device).squeeze(0)
            
        # 버퍼가 가득 차면 훈련 시작
        if buffer.ptr == N_STEPS:
            with th.no_grad():
                next_obs_tensor = th.from_numpy(next_obs).float().to(device)
                _, last_value, _ = agent.network(next_obs_tensor, hidden_state)
            
            buffer.compute_returns_and_advantages(last_value.cpu().item(), done)
            agent.update(buffer)
            buffer.reset()

    print(f"훈련 완료! (소요 시간: {time.time() - start_time:.2f}초)")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title("학습 곡선")
    plt.xlabel("에피소드")
    plt.ylabel("총 보상")
    plt.show()

    env.close()