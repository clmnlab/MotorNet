import os
import motornet as mn
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.utils import resample
import torch as th


base_dir = os.path.join(os.path.expanduser('~'),'Documents','Data','MotorNet')

def window_average(x, w=10):
    rows = int(np.size(x)/w) # round to (floor) int
    cols = w
    return x[0:w*rows].reshape((rows,cols)).mean(axis=1)

def load_env(task,cfg=None,dT=None):
    # also get K and B
    if cfg is None:

        name = 'env'

        action_noise         = 1e-4
        proprioception_noise = 1e-3
        vision_noise         = 1e-4
        vision_delay         = 0.07
        proprioception_delay = 0.02

        # Define task and the effector
        effector = mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle())

        max_ep_duration = 1
    else:
        name = cfg['name']
        # effector
        muscle_name = cfg['effector']['muscle']['name']
        #timestep = cfg['effector']['dt']
        if dT is None:
            timestep = cfg['dt']
        else:
            timestep = dT
        cfg['dt'] = timestep
        muscle = getattr(mn.muscle,muscle_name)()
        effector = mn.effector.RigidTendonArm26(muscle=muscle,timestep=timestep) 


        # delay
        proprioception_delay = cfg['proprioception_delay']*cfg['dt']
        vision_delay = cfg['vision_delay']*cfg['dt']

        # noise
        action_noise = cfg['action_noise'][0]
        proprioception_noise = cfg['proprioception_noise'][0]
        vision_noise = cfg['vision_noise'][0]

        # initialize environment
        max_ep_duration = cfg['max_ep_duration']


    env = task(effector=effector,max_ep_duration=max_ep_duration,name=name,
               action_noise=action_noise,proprioception_noise=proprioception_noise,
               vision_noise=vision_noise,proprioception_delay=proprioception_delay,
               vision_delay=vision_delay)

    return env


def load_policy(n_input,n_output,weight_file=None,phase='growing_up',freeze_output_layer=False,freeze_input_layer=False,n_hidden=128):

    import torch as th
    device = th.device("cpu")
    
    from policy import Policy
    policy = Policy(n_input, n_hidden, n_output, device=device, 
                    freeze_output_layer=freeze_output_layer, freeze_input_layer=freeze_input_layer)
    
    if weight_file is not None:
        policy.load_state_dict(th.load(weight_file,map_location=device))

    if phase=='growing_up':
        optimizer = th.optim.Adam(policy.parameters(), lr=3e-3)
        scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer,  gamma=0.9999)
    else:
        optimizer = th.optim.SGD(policy.parameters(), lr=5e-3)
        scheduler = None
        
    return policy, optimizer, scheduler


def load_stuff(cfg_file,weight_file,phase='growing_up',freeze_output_layer=False, freeze_input_layer=False,n_hidden=128):
    # also get K and B
    import json
    from task import CentreOutFF

    # load configuration
    cfg = None
    if cfg_file is not None:
        cfg = json.load(open(cfg_file, 'r'))
    env = load_env(CentreOutFF, cfg)

    n_input = env.observation_space.shape[0]
    n_output = env.n_muscles

    # load policy
    policy, optimizer, scheduler = load_policy(n_input,n_output,weight_file=weight_file,phase=phase,
                                               freeze_output_layer=freeze_output_layer, freeze_input_layer=freeze_input_layer,n_hidden=n_hidden)

    return env, policy, optimizer, scheduler
        

def calculate_angles_between_vectors(vel, tg, xy):
    """
    Calculate angles between vectors X2 and X3.

    Parameters:
    - vel (numpy.ndarray): Velocity array.
    - tg (numpy.ndarray): Tg array.
    - xy (numpy.ndarray): Xy array.

    Returns:
    - angles (numpy.ndarray): An array of angles in degrees between vectors X2 and X3.
    """

    tg = np.array(tg)
    xy = np.array(xy)
    vel = np.array(vel)
    
    # Compute the magnitude of velocity and find the index to the maximum velocity
    vel_norm = np.linalg.norm(vel, axis=-1)
    idx = np.argmax(vel_norm, axis=1)

    # Calculate vectors X2 and X3
    X2 = tg[:,-1,:]
    X1 = xy[:,25,:]
    X3 = xy[np.arange(xy.shape[0]), idx, :]

    X2 = X2 - X1
    X3 = X3 - X1
    
    cross_product = np.cross(X3, X2)
    # Calculate the sign of the angle
    sign = np.sign(cross_product)

    # Calculate the angles in degrees
    angles = sign*np.degrees(np.arccos(np.sum(X2 * X3, axis=1) / (1e-8+np.linalg.norm(X2, axis=1) * np.linalg.norm(X3, axis=1))))

    return angles

def calculate_lateral_deviation(xy, tg, vel=None):
    """
    Calculate the lateral deviation of trajectory xy from the line connecting X1 and X2.

    Parameters:
    - tg (numpy.ndarray): Tg array.
    - xy (numpy.ndarray): Xy array.

    Returns:
    - deviation (numpy.ndarray): An array of lateral deviations.
    """
    tg = np.array(tg)
    xy = np.array(xy)

    # Calculate vectors X2 and X1
    X2 = tg[:,-1,:]
    X1 = xy[:,25,:]

    # Calculate the vector representing the line connecting X1 to X2
    line_vector = X2 - X1
    line_vector2 = np.tile(line_vector[:,None,:],(1,xy.shape[1],1))

    # Calculate the vector representing the difference between xy and X1
    trajectory_vector = xy - X1[:,None,:]

    projection = np.sum(line_vector2 * trajectory_vector, axis=-1)/np.sum(line_vector2 * line_vector2, axis=-1)
    projection = line_vector2 * projection[:,:,np.newaxis]

    lateral_dev = np.linalg.norm(trajectory_vector - projection,axis=2)

    idx = np.argmax(lateral_dev,axis=1)

    max_laterl_dev = lateral_dev[np.arange(idx.shape[0]), idx]

    init = projection[np.arange(idx.shape[0]),idx,:]
    init = init+X1

    endp = xy[np.arange(idx.shape[0]),idx,:]


    cross_product = np.cross(endp-X1, X2-X1)
    # Calculate the sign of the angle
    sign = np.sign(cross_product)


    opt={'lateral_dev':np.mean(lateral_dev,axis=-1),
         'max_lateral_dev':max_laterl_dev,
         'lateral_vel':None}
    # speed 
    if vel is not None:
        vel = np.array(vel)
        projection = np.sum(line_vector2 * vel, axis=-1)/np.sum(line_vector2 * line_vector2, axis=-1)
        projection = line_vector2 * projection[:,:,np.newaxis]
        lateral_vel = np.linalg.norm(vel - projection,axis=2)
        opt['lateral_vel'] = np.mean(lateral_vel,axis=-1)

    return sign*max_laterl_dev, init, endp, opt

def optimize_channel(cfg_file,weight_file):

    def lat_loss(theta):
        from model import test
        data = test(cfg_file,weight_file,is_channel=True,K=theta[0],B=theta[1])
        _, _, _, opt = calculate_lateral_deviation(data['xy'], data['tg'], data['vel'])
        return np.mean(opt['max_lateral_dev'])

    # find K and B such that max lateral deviation is minimized...
    loss_before = lat_loss([0,0])

    theta0 = [180,-2]
    theta = minimize(lat_loss,theta0,method='Nelder-Mead',options={'maxiter':10000,'disp':False})
    loss_after = lat_loss(theta.x)

    print(f'loss before: {loss_before}')
    print(f'loss after: {loss_after}')

    return theta.x


def sweep_loss():
    loss_weights = np.array([1e+3,   # position
                                 1e+5,   # jerk
                                 1e-1,   # muscle
                                 1e-5,   # muscle_derivative
                                 3e-5,   # hidden 
                                 2e-2,   # hidden_derivative
                                 0])     # hidden_jerk
    md_w = np.logspace(start=-6,stop=-1,num=3)
    h_w = np.logspace(start=-5,stop=-2,num=3)
    hd_w = np.logspace(start=-3,stop=-1,num=3)

    mhh_w = np.array(list(product(md_w,h_w,hd_w)))

    iter_list = range(20)
    num_processes = len(iter_list)

    lw = [loss_weights.copy() for _ in range(len(mhh_w))]

    for idx,mhhw in enumerate(mhh_w):
        lw[idx][3] = mhhw[0]
        lw[idx][4] = mhhw[1]
        lw[idx][5] = mhhw[2]
    return lw


# cal_loss was originally in model.py
def calc_loss(data, loss_weight=None):

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

  # loss['hidden'] = th.mean(th.sum(th.square(data['all_hidden']), dim=-1))
  # loss['hidden_derivative'] = th.mean(th.sum(th.square(th.diff(data['all_hidden'], n=1, dim=1)), dim=-1))
  # loss['hidden_jerk'] = th.mean(th.sum(th.square(th.diff(data['all_hidden'], n=3, dim=1)), dim=-1))
  
  

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





def bootstrap_ci(data, n_boot=1000, ci=95):
    """
    Calculate the bootstrap confidence interval.
    """
    boot_means = []
    for _ in range(n_boot):
        boot_sample = resample(data)
        boot_means.append(np.mean(boot_sample))
    
    lower = np.percentile(boot_means, (100-ci)/2)
    upper = np.percentile(boot_means, 100-(100-ci)/2)
    
    return (lower, upper)

def ci_func(values):
    """
    Compute the lower and upper confidence interval from bootstrap sampling.
    """
    ci_low, ci_high = bootstrap_ci(values)
    return ci_low, ci_high



# Normalize the 'us' values
def normalize_phase_data(df):
    scale = []
    normalized_data = df.copy()
    for model in df['mn'].unique():
        model_data = df[df['mn'] == model]
        NF1_value = model_data[model_data['phase'] == 'NF1']['us'].values[0]
        FF1_value = model_data[model_data['phase'] == 'FF1']['us'].values[0]
        scaling_factor = FF1_value - NF1_value
        scale.append(scaling_factor)    
        # Apply normalization
        normalized_data.loc[normalized_data['mn'] == model, 'us'] = (
            model_data['us'] - NF1_value) / scaling_factor

    return normalized_data,scale

class modelLoss():
    def __init__(self):
        #self.n_param = 5
        pass
    def predict(self,theta=None):
        if theta is None:
            theta = self.theta
        #pred = theta[0]*np.exp(-theta[1]*self.x**2)+ theta[2]

        # pred = np.exp(theta[0])*np.exp(-np.exp(theta[1])*self.x**3) +\
        #       np.exp(theta[2])*np.exp(-np.exp(theta[3])*self.x**2) +\
        #         np.exp(theta[4])*np.exp(-np.exp(theta[5])*self.x**1) + theta[6] 
        
        #pred = np.exp(theta[0])*np.exp(-np.exp(theta[1])*self.x**3) +\
        #        np.exp(theta[2])*np.exp(-np.exp(theta[3])*self.x**1) + theta[4] 
        

        # pred = np.exp(theta[0])*np.exp(-np.exp(theta[1])*self.x**=3) +\
        #         np.exp(theta[2])*np.exp(-np.exp(theta[3])*self.x**1) + theta[4] 
        pred = np.exp(theta[0])*np.exp(-np.exp(theta[1])*self.x**1) +\
            + theta[2] 


        return pred
    def fro(self,theta,loss,lam=None):
        pred = self.predict(theta)
        return np.linalg.norm(loss-pred)+lam*np.sum(np.abs(theta))
    def fit(self,loss,lam=None,theta0=None):
        if lam is None:
            lam = 0
        
        self.x = np.arange(len(loss))

        if theta0 is None:
            #theta0 = [-4.87692324e-05, -9.65588447e-06,  2.65996536e+00, -4.56851626e+00,2.18185688e+01]
            theta0 = [-1, -1,22]
        theta = minimize(self.fro,theta0,args=(loss,lam),method='Nelder-Mead',options={'maxiter':10000,'disp':False})
        self.theta = theta.x
        self.success = theta.success
    def find_x(self,loss_thresh):
        pred = self.predict()
        idx = np.where(pred<=loss_thresh)[0]
        return idx[0]
    def get_rate(self):
        return np.exp(self.theta[1])
    
def get_initial_loss(loss):
    
    T = pd.DataFrame()

    # get initial loss
    data = {'NF1':[],'FF1':[],'NF2':[],'FF2':[]}
    for p in list(data.keys()):
        index=0
        if p=='NF1' or p=='NF2':
            index=-1
        data[p] = list(np.array(loss[p])[:,index])

    T = create_dataframe(data)
    T['feature'] = 'init'
    
    return T

def get_rate(loss,w=10,check_fit=False):
    loss2 = deepcopy(loss)
    if w>1:
        for phase in loss2.keys():
            loss2[phase] = [window_average(np.array(l), w) for l in loss2[phase]]
    

    T = pd.DataFrame()

    # Fit data
    data = {'FF1':[],'FF2':[]} # this will contain the rate
    pred = {'FF1':[],'FF2':[]} # just for checking the fit
    
    for m in range(len(loss2['FF1'])):
        for _,phase in enumerate(data.keys()):
            l = loss2[phase][m]

            model = modelLoss()

            theta0=[np.log(l[0]),np.log(0.004),l[-1]]
            model.fit(l,lam=0.0,theta0=theta0)

            pred[phase].append(model.predict())
            data[phase].append(model.get_rate())
    
    T = create_dataframe(data)
    T['feature'] = 'rate'

    # Check the fits
    if check_fit:
        _,ax = plt.subplots(1,2,figsize=(6,5))
        ax[0].plot(np.mean(loss2['FF1'],axis=0),linestyle='-',color='b',label='data')
        ax[0].plot(np.mean(pred['FF1'],axis=0),linestyle='--',color='r',label='pred')
        ax[0].legend()

        ax[1].plot(np.mean(loss2['FF2'],axis=0),linestyle='-',color='b',label='data')
        ax[1].plot(np.mean(pred['FF2'],axis=0),linestyle='--',color='r',label='pred')
        ax[1].legend()
        plt.show()

    return T


def create_dataframe(idx):
    data = []
    for p in list(idx.keys()):
        val = idx[p]
        data.extend([
            {'mn': i + 1, 'phase': p, 'value': v}
            for i, v in enumerate(val)
        ])
    return pd.DataFrame(data)



def calc_loss_batch(data, loss_weight=None):   # ChatGPT created this function
    """
    data: dict of tensors, each tensor shape (batch, T, feature_dim?) 또는 (batch, T) 등.
      예: data['xy'] shape (batch, T, xy_dim), data['tg'] same shape.
    loss_weight: None 또는 array-like of length 7, 순서:
      ['position', 'jerk', 'muscle', 'muscle_derivative', 'hidden', 'hidden_derivative', 'hidden_jerk']
      각 항의 배치별 스칼라 손실에 곱할 가중치.
    Returns:
      overall_loss: 스칼라 텐서 (batch-wise weighted sum을 batch 평균)
      loss_terms: dict of per-term weighted 손실 텐서, shape (batch,)
    """

    # 1) 우선 각 항의 "비가중치" 손실을 batch-wise로 계산: shape (batch,)
    loss = {}

    # position: 시간축에 대해 |xy - tg| 의 L1 혹은 L2 등을 쓸 수 있는데, 여기선 L1:
    # data['xy'], data['tg'] shape (batch, T, dim)
    # abs difference: (batch, T, dim) -> sum over dim -> (batch, T) -> mean over time dim=1 -> (batch,)
    loss['position'] = th.mean(
        th.sum(th.abs(data['xy'] - data['tg']), dim=-1),
        dim=1
    )  # shape (batch,)

    # jerk: 속도의 2차 차분 제곱합
    # data['vel'] shape (batch, T, vel_dim)
    # th.diff(..., n=2, dim=1) -> shape (batch, T-2, vel_dim)
    # square, sum over feature dim -> (batch, T-2), then mean over time dim=1 -> (batch,)
    vel = data['vel']
    # 주의: T 길이가 2보다 커야 함
    if vel.size(1) >= 3:
        jerk_tensor = th.diff(vel, n=2, dim=1)  # (batch, T-2, vel_dim)
        loss['jerk'] = th.mean(
            th.sum(jerk_tensor.square(), dim=-1),
            dim=1
        )  # (batch,)
    else:
        # 너무 짧으면 0으로 처리
        loss['jerk'] = th.zeros(data['vel'].shape[0], device=data['vel'].device)

    # muscle: all_force shape (batch, T, force_dim). 예: sum over feature, mean over time:
    loss['muscle'] = th.mean(
        th.sum(data['all_force'], dim=-1),
        dim=1
    )  # (batch,)

    # muscle_derivative: 1차 차분 제곱합
    force = data['all_force']
    if force.size(1) >= 2:
        d_force = th.diff(force, n=1, dim=1)  # (batch, T-1, force_dim)
        loss['muscle_derivative'] = th.mean(
            th.sum(d_force.square(), dim=-1),
            dim=1
        )  # (batch,)
    else:
        loss['muscle_derivative'] = th.zeros(data['all_force'].shape[0], device=data['all_force'].device)

    # hidden: all_hidden shape (batch, T, hidden_dim)
    # 예: 제곱합 후 시간 평균
    loss['hidden'] = th.mean(
        th.sum(data['all_hidden'].square(), dim=-1),
        dim=1
    )  # (batch,)

    # hidden_derivative: 1차 차분
    h = data['all_hidden']
    if h.size(1) >= 2:
        dh = th.diff(h, n=1, dim=1)  # (batch, T-1, hidden_dim)
        loss['hidden_derivative'] = th.mean(
            th.sum(dh.square(), dim=-1),
            dim=1
        )  # (batch,)
    else:
        loss['hidden_derivative'] = th.zeros(data['all_hidden'].shape[0], device=data['all_hidden'].device)

    # hidden_jerk: 3차 차분
    if h.size(1) >= 4:
        dh3 = th.diff(h, n=3, dim=1)  # (batch, T-3, hidden_dim)
        loss['hidden_jerk'] = th.mean(
            th.sum(dh3.square(), dim=-1),
            dim=1
        )  # (batch,)
    else:
        loss['hidden_jerk'] = th.zeros(data['all_hidden'].shape[0], device=data['all_hidden'].device)

    # 2) 가중치 준비: numpy array나 list 형태로 들어오면 tensor로 변환
    if loss_weight is None:
        # 기본 weight: position, jerk, muscle, muscle_derivative, hidden, hidden_derivative, hidden_jerk
        # 예시 값: np.array([1e+3,1e+5,1e-1,3e-4,1e-5,1e-3,0])
        w = th.tensor([1e+3, 1e+5, 1e-1, 3e-4, 1e-5, 1e-3, 0.0],
                      device=next(iter(loss.values())).device, dtype=next(iter(loss.values())).dtype)
    else:
        # loss_weight이 list/np.array 등일 때
        w = th.tensor(loss_weight, device=next(iter(loss.values())).device,
                      dtype=next(iter(loss.values())).dtype)
        if w.numel() != 7:
            raise ValueError(f"loss_weight must have length 7, got {w.numel()}")

    # 3) 항목별 weighted loss: shape (batch,)
    loss_weighted = {
        'position': w[0] * loss['position'],
        'jerk': w[1] * loss['jerk'],
        'muscle': w[2] * loss['muscle'],
        'muscle_derivative': w[3] * loss['muscle_derivative'],
        'hidden': w[4] * loss['hidden'],
        'hidden_derivative': w[5] * loss['hidden_derivative'],
        'hidden_jerk': w[6] * loss['hidden_jerk'],
    }

    # 4) overall loss: 배치별 합산 후 평균
    # position 항은 원래 코드에서 th.mean(loss_weighted['position'])을 했는데,
    # 이미 loss_weighted['position']은 shape (batch,), 따라서
    # overall_loss = torch.mean(sum over keys of loss_weighted[key]) 형태로 계산.
    total_per_sample = None
    for key, lw in loss_weighted.items():
        if total_per_sample is None:
            total_per_sample = lw
        else:
            total_per_sample = total_per_sample + lw
    # total_per_sample: shape (batch,)
    overall_loss = th.mean(total_per_sample)  # scalar

    return overall_loss, loss_weighted  # loss_weighted 값은 (batch,) 텐서들



def run_rollout(env,agent,batch_size=1, catch_trial_perc=50,condition='train', # ff_coefficient = 0.
                ff_coefficient=0., is_channel=False,detach=False,calc_endpoint_force=False, go_cue_random=None,
                disturb_hidden=False, t_disturb_hidden=0.15, d_hidden=None, seed=None):
  
    device = agent.device
    h = agent.policy_net.init_hidden(batch_size = batch_size).to(device)
    obs, info = env.reset(condition=condition, catch_trial_perc=catch_trial_perc, ff_coefficient=ff_coefficient, options={'batch_size': batch_size}, 
                          is_channel=is_channel,calc_endpoint_force=calc_endpoint_force, go_cue_random=go_cue_random, seed = seed)

    obs = obs.to(device)
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

        action, h = agent.policy_net(obs,h)

        obs, terminated, info = env.step(action=action.to('cpu'))
        obs = obs.to(device)
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