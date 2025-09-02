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
import buffer
from importlib import reload
reload(agents)
reload(utils)
reload(task)
reload(buffer)
from utils import load_env, load_policy, calc_loss, run_rollout
from task import CentreOutFFGym
from buffer import ReplayBufferOnPolicy
from agents import SLAgent, GRUPPOAgent

import time


def run_experiment(name='gruppo_agent', device='cuda', load_path=None, config = 'params.json', condition = 'train', ff_coeff = 0):
    
    # --- 하이퍼파라미터 ---
    TOTAL_TIMESTEPS = 2000000
    N_STEPS = 100  # 데이터 수집 스텝 수
    HIDDEN_DIM = 128
    save_dir = Path.cwd() / 'results' / name
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving in the following directory: {save_dir}")
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
    agent = GRUPPOAgent(env, hidden_dim=HIDDEN_DIM,  lr=5e-4, device=device)
        # --- 저장된 모델이 있으면 로드 ---
    if load_path is not None:
        load_path = Path(load_path)
        if load_path.exists():
            
            print(f"loading a saved model: {load_path}")
            agent.load(load_path)
        else:
            print(f"Model does not exist: {load_path}")
            
    buffer = ReplayBufferOnPolicy(env.observation_space.shape[0], env.action_space.shape[0], HIDDEN_DIM, train_params['batch_size'])

    print(f"훈련 시작 (총 {TOTAL_TIMESTEPS} 스텝) on {device}...")
    start_time = time.time()
    
    obs, _ = env.reset(options={'batch_size': train_params['batch_size']})
    hidden_state = agent.network.init_hidden(train_params['batch_size'])
    
    episode_rewards = []
    current_episode_reward = 0
    episode_idx = 0
    for step in range(TOTAL_TIMESTEPS):
        # obs_tensor = th.from_numpy(obs).float().to(device)
        action, action_raw, value, log_prob, next_hidden_state = agent.select_action(obs, hidden_state)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        # print(terminated[0], step)
        buffer.add(obs, action_raw, action, reward, terminated, log_prob, value, hidden_state)
        current_episode_reward += float(np.mean(reward))
        
        obs = next_obs
        hidden_state = next_hidden_state
        # done = terminated or truncated, update the agent
        if terminated[0]:
            with th.no_grad():
                next_obs_tensor = th.from_numpy(next_obs).float().to(device)
                _, last_value, _ = agent.network(next_obs_tensor, hidden_state)
            
            buffer.compute_returns_and_advantages(last_value.cpu(), terminated[0])
            ploss, vloss, eloss = agent.update(buffer)
            buffer.reset()
            episode_rewards.append(current_episode_reward)
            episode_idx += 1
            print(f"에피소드 {episode_idx} 완료, 보상, loss: {current_episode_reward:.2f}, {ploss:.4f}, {vloss:.4f}, {eloss:.4f}")
            # if len(episode_rewards) % 5 == 0:1234
                # print(f"스텝 {step+1}: 최근 105 에피소드 평균 보상, loss: {np.mean(episode_rewards[-5:]):.2f}, {ploss:.4f}, {vloss:.4f}, {eloss:.4f}")
            current_episode_reward = 0
            obs, _ = env.reset(options={'batch_size': train_params['batch_size']})
            hidden_state = agent.network.init_hidden(train_params['batch_size'])
        if (episode_idx) % 10 == 0:
            agent.save(save_dir/f'agent_epi{episode_idx}.pth')
            with open(save_dir / 'episode_rewards.json', 'w') as f:
                json.dump({'episode_rewards': episode_rewards}, f, indent=4)

    print(f"훈련 완료! (소요 시간: {time.time() - start_time:.2f}초)")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title("학습 곡선")
    plt.xlabel("에피소드")
    plt.ylabel("총 보상")
    plt.show()

    env.close()
    
if __name__ == '__main__':
    # run_experiment(name='gruppo_agent2', device='cuda', load_path='results/gruppo_agent2/agent_500000.pth', config='params.json', condition='train',ff_coeff=0.0)
    # run_experiment(name='gruppo_agent6', device='cuda', load_path='results/gruppo_agent5/agent_epi13870.pth', config='params.json', condition='train',ff_coeff=0.0)
    # run_experiment(name='gruppo_agent12', device='cuda', load_path='results/gruppo_agent12/agent_epi4500.pth', config='params.json', condition='train',ff_coeff=0.0)
    run_experiment(name='gruppo_agent13', device='cuda', load_path=None, config='params.json', condition='train',ff_coeff=0.0)
    