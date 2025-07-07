
import os
import sys
import json
import numpy as np
import torch as th
import matplotlib.pyplot as plt
import motornet as mn
import utils
import agents
import task
from pathlib import Path
from tqdm import tqdm
import argparse
import json    

from importlib import reload
reload(agents)
reload(utils)
reload(task)
from utils import load_env, load_policy, calc_loss, run_rollout
from model import run_episode
from task import CentreOutFF
from agents import SLAgent
# from sb3_contrib import RecurrentPPO
# ========================================================================================
# 2. 메인 훈련 및 적응 함수
# ========================================================================================
def run_experiment(name='exp_train', device='cpu', load_path=None):
    """
    초기 훈련과 적응 학습을 순차적으로 진행하는 메인 함수.
    """

    action_noise         = 1e-4
    proprioception_noise = 1e-3
    vision_noise         = 1e-4
    vision_delay         = 0.07
    proprioception_delay = 0.02
    max_ep_duration = 1.0
    batch_size = 128
    n_batch = 20000
    losses = []
    interval = 100

    # --- 디렉토리 설정 ---
    save_dir = Path.cwd() / 'results' / name
    os.makedirs(save_dir, exist_ok=True)
    print(f"결과는 다음 디렉토리에 저장됩니다: {save_dir}")
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")




    # Define task and the effector
    effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())

    env = CentreOutFF(effector=effector,max_ep_duration=max_ep_duration,name=name,
        action_noise=action_noise,proprioception_noise=proprioception_noise,
        vision_noise=vision_noise,proprioception_delay=proprioception_delay,
        vision_delay=vision_delay)
    n_input = env.observation_space.shape[0]
    n_output = env.n_muscles


    agent = SLAgent(obs_dim=n_input, action_dim=n_output, batch_size = batch_size, device=device)

    # --- 저장된 모델이 있으면 로드 ---
    if load_path is not None:
        load_path = Path(load_path)
        if load_path.exists():
            print(f"모델을 불러옵니다: {load_path}")
            agent.load(load_path)
        else:
            print(f"지정한 모델 경로가 존재하지 않습니다: {load_path}")

    for batch in range(n_batch):
        data = run_rollout(env, agent, batch_size=batch_size, device=device)
        loss = agent.update(data)
        losses.append(loss.item())

        if batch % interval == 0:
            avg = sum(losses[-interval:]) / len(losses[-interval:])  # Python float 평균
            print(f"Batch {batch}, Loss: {avg:.6f}")
            agent.save(save_dir / f'agent_{batch}.pth')
    with open(save_dir / 'losses.json', 'w') as f:
        json.dump({'losses': losses}, f, indent=4)
# ========================================================================================
# 4. 스크립트 실행
# ========================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='exp_train', help='실험 이름')
    parser.add_argument('--device', type=str, default='cpu', help='사용할 디바이스: cpu 또는 cuda')
    parser.add_argument('--load', type=str, default=None, help='불러올 모델의 경로 (예: results/exp_train/agent_1000.pth)')
    args = parser.parse_args()
    run_experiment(name=args.name, device=args.device, load_path=args.load)
    print("실험이 완료되었습니다.")
    # --- 결과 저장 --- 

    
    
    

