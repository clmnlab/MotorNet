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

def run_episode(env,policy,batch_size=1, catch_trial_perc=50,condition='train',
                ff_coefficient=0., is_channel=False,detach=False,calc_endpoint_force=False, go_cue_random=None,
                disturb_hidden=False, t_disturb_hidden=0.15, d_hidden=None, seed=None):  # ff_coefficient = 0.
  
  h = policy.init_hidden(batch_size=batch_size)
  obs, info = env.reset(condition=condition, catch_trial_perc=catch_trial_perc, ff_coefficient=ff_coefficient, options={'batch_size': batch_size}, 
                        is_channel=is_channel,calc_endpoint_force=calc_endpoint_force, go_cue_random=go_cue_random, seed = seed)
  terminated = False

  # Initialize a dictionary to store lists
  data = {
    'xy': [],
    'tg': [],
    'vel': [],
    'all_action': [],
    'all_hidden': [],
    'all_muscle': [],
    'all_force': [],
    'endpoint_load': [],
    'endpoint_force': []
  }

  while not terminated:

    # add disturn hidden activity
    if disturb_hidden:
      if np.abs(env.elapsed-t_disturb_hidden)<1e-3:
        #print('DONE!!!!')
        dh = d_hidden.repeat(1,batch_size,1)
        h += dh

    action, h = policy(obs,h)
    obs, terminated, info = env.step(action=action)
    data['all_hidden'].append(h[0, :, None, :])
    data['all_muscle'].append(info['states']['muscle'][:, 0, None, :])
    data['all_force'].append(info['states']['muscle'][:, -1, None, :])
    data['xy'].append(info["states"]["fingertip"][:, None, :])
    data['tg'].append(info["goal"][:, None, :])
    data['vel'].append(info["states"]["cartesian"][:, None, 2:])  # velocity
    data['all_action'].append(action[:, None, :])
    data['endpoint_load'].append(info['endpoint_load'][:, None, :])
    data['endpoint_force'].append(info['endpoint_force'][:, None, :])
      

  # Concatenate the lists
  for key in data:
    data[key] = th.cat(data[key], axis=1)

  if detach:
    # Detach tensors if needed
    for key in data:
        data[key] = th.detach(data[key])

  return data