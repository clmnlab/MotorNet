import numpy as np
from copy import deepcopy


import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel



def gsog(X):
    """
    Gram-Schmidt orthogonalization

    Parameters:
    - X (ndarray): Input matrix of size (d, n)

    Returns:
    - Q (ndarray): Orthogonalized matrix of size (d, m)
    - R (ndarray): Upper triangular matrix of size (m, n)
    """

    d, n = X.shape
    m = min(d, n)

    R = np.eye(m, n)
    Q = np.zeros((d, m))
    D = np.zeros(m)

    for i in range(m):
        R[0:i, i] = np.dot(np.multiply(Q[:, 0:i], 1 / D[0:i]).T, X[:, i])
        Q[:, i] = X[:, i] - np.dot(Q[:, 0:i], R[0:i, i])
        D[i] = np.dot(Q[:, i], Q[:, i])

    R[:, m:n] = np.dot(np.multiply(Q, 1 / D).T, X[:, m:n])

    return Q, R

def build_tdr(X,N):
    """
    Perform TDR analyses as in Sun, O'Shea et al, 2021.

    Parameters:
    - X (ndarray): Behavioral variables before learning (C by M matrix)
    - N (ndarray): Condition-averaged, centered neural activity before learning (C by N matrix)

    Returns:
    - beta_n2b_orth (ndarray): Matrix of orthogonalized coefficients projecting neural activity to TDR axes
    """
    # Get ready the design matrix (behavioral variables + intercept).
    X = np.hstack((X,np.ones((X.shape[0],1))))

    # Regress neural data against the design matrix and compute the regression coefficients.
    beta_b2n = np.linalg.pinv(X) @ N
    #beta_b2n = np.linalg.lstsq(X, N, rcond=None)[0]

    # Compute the TDR axes.
    beta_n2b = np.linalg.pinv(beta_b2n)
    beta_n2b = beta_n2b[:, :2]

    # Orthogonalize the TDR axes before projection.
    beta_n2b_orth = gsog(beta_n2b)[0]

    return beta_n2b_orth

def project_onto_map(data,map,remove_mean=True,mean_all=True):
    """
    Returns:
    - data_p (ndarray): Matrix of neural state coordinates on orthogonalized TDR axes
    """
    data_p = deepcopy(data)
    # remove the mean

    if mean_all==True:
        combined_N = np.vstack(data)
        mean_N = np.mean(combined_N, axis=0)
    else:
        mean_N = np.mean(data[0], axis=0)

    if remove_mean==False:
        mean_N = np.zeros_like(mean_N)

    for i in range(len(data)):
        data_p[i] = (data[i]-mean_N) @ map
    return data_p

def orth_wrt_map(us, map):
    us_orth = us.copy()  # Start with the original vector
    for i in range(map.shape[1]):  # Iterate over each column of the map
        # Project us onto the current column and subtract the projection from us_orth
        us_orth = us_orth - np.dot(map[:,i], us_orth)/np.linalg.norm(map[:,i])**2 * map[:,i][:,None]
    us_orth_norm = us_orth / np.linalg.norm(us_orth)  # Normalize the orthogonal vector
    return us_orth, us_orth_norm


## Gemin-specific TDR analysis and visualization: Gemini generates
def analyze_tdr(hidden_states_data, target_coords_data):
    """
    주어진 hidden state 데이터에 TDR을 적용하여,
    타겟 위치를 예측하는 2D 신경 궤적을 분석하고 시각화합니다.

    Args:
        hidden_states_data (np.ndarray): 분석할 은닉 상태 데이터.
                                         형태: (batch_size, time_steps, hidden_dims)
        target_coords_data (np.ndarray): 각 trial의 목표 지점 (x, y) 좌표.
                                          형태: (batch_size, 2)
    """
    print("Targeted Dimensionality Reduction (TDR) 분석 시작...")

    # 1. 데이터 형태 확인 및 준비
    if hidden_states_data.ndim != 3:
        raise ValueError("hidden_states_data는 (batch_size, time_steps, hidden_dims) 형태여야 합니다.")
    if target_coords_data.ndim != 2 or target_coords_data.shape[1] != 2:
        raise ValueError("target_coords_data는 (batch_size, 2) 형태여야 합니다.")
    
    batch_size, time_steps, hidden_dims = hidden_states_data.shape
    print(f"데이터 형태: Batch={batch_size}, Time Steps={time_steps}, Hidden Dims={hidden_dims}")

    # 2. 회귀 모델 학습
    # TDR을 위해, 특정 시점의 신경 활동으로 타겟 위치를 예측하는 모델을 학습합니다.
    # 여기서는 각 trial의 중간 시점(t=50)의 은닉 상태를 사용합니다.
    X_train = hidden_states_data[:, 50, :]  # (batch_size, hidden_dims)
    y_train = target_coords_data             # (batch_size, 2)

    print("가우시안 프로세스 회귀 모델 학습 중...")
    # 논문에서 사용한 것과 유사한 커널 설정
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X_train, y_train)
    print("모델 학습 완료.")

    # 3. 전체 궤적을 TDR 공간으로 "투영" (예측)
    # 모든 시간 스텝의 은닉 상태를 2D로 펼쳐서 모델의 예측값을 얻습니다.
    # 이 예측값 자체가 TDR로 축소된 2차원 공간이 됩니다.
    all_hidden_states_reshaped = hidden_states_data.reshape(-1, hidden_dims)
    tdr_trajectory_2d = gpr.predict(all_hidden_states_reshaped)
    
    # 원래 데이터 형태로 다시 복원: (batch * steps, 2) -> (batch, steps, 2)
    tdr_trajectory_2d = tdr_trajectory_2d.reshape(batch_size, time_steps, 2)

    # 4. 시각화
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 8방향 타겟에 해당하는 색상 맵
    target_colors = plt.cm.hsv(np.linspace(0, 1, 8))

    # 각 trial(궤적)을 그림
    for i in range(batch_size):
        trajectory = tdr_trajectory_2d[i, :, :]
        
        # 각 trial은 특정 타겟 방향에 해당하므로, 그에 맞는 색상을 지정
        # (8방향 타겟을 반복했다고 가정)
        color = target_colors[i % 8]
        
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.7)
        # 시작점 (작은 점)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color=color, s=20)
        # 끝점 (큰 점)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, s=150, ec='w')

    ax.set_title("Neural Trajectory in Target-Predictive Subspace (TDR)", fontsize=16)
    ax.set_xlabel("TDR Dimension 1 (Predicted X)", fontsize=12)
    ax.set_ylabel("TDR Dimension 2 (Predicted Y)", fontsize=12)
    ax.grid(True)
    ax.axis('equal') # 축의 스케일을 동일하게 맞춤
    plt.show()
