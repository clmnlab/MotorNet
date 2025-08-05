import torch as th
import torch.nn as nn
from torch.distributions import Normal

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class GRUPolicy(th.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device, freeze_output_layer=False, 
                 learn_h0=True, freeze_input_layer=False,freeze_bias_hidden=False,freeze_h0=False):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        
        self.gru = th.nn.GRU(input_dim, hidden_dim, 1, batch_first=True)
        self.fc = th.nn.Linear(hidden_dim, output_dim)
        self.sigmoid = th.nn.Sigmoid()
        
        if freeze_output_layer:
            for param in self.fc.parameters():
                param.requires_grad = False
        if freeze_input_layer:
            for name, param in self.gru.named_parameters():
                if name == "weight_ih_l0" or name == "bias_ih_l0":
                    param.requires_grad = False

        if freeze_bias_hidden:
            for name, param in self.gru.named_parameters():
                if name == "bias_hh_l0":
                    param.requires_grad = False

        # the default initialization in torch isn't ideal
        for name, param in self.named_parameters():
            if name == "gru.weight_ih_l0":
                th.nn.init.xavier_uniform_(param)
            elif name == "gru.weight_hh_l0":
                th.nn.init.orthogonal_(param)
            elif name == "gru.bias_ih_l0":
                th.nn.init.zeros_(param)
            elif name == "gru.bias_hh_l0":
                th.nn.init.zeros_(param)
            elif name == "fc.weight":
                th.nn.init.xavier_uniform_(param)
            elif name == "fc.bias":
                th.nn.init.constant_(param, -5.)
            else:
                raise ValueError
        if learn_h0:
            self.h0 = th.nn.Parameter(th.zeros(self.n_layers, 1, hidden_dim), requires_grad=True)

        if freeze_h0:
            for name, param in self.named_parameters():
                if name == "h0":
                    param.requires_grad = False
        
        self.to(device)

    def forward(self, x, h0):

        # TODO
        # Here I can add noise to h0 before applying
        y, h = self.gru(x[:, None, :], h0)
        #hidden_noise         = 1e-3
        u = self.sigmoid(self.fc(y)).squeeze(dim=1)
        return u, h
    
    def init_hidden(self, batch_size):
        
        if hasattr(self, 'h0'):
            hidden = self.h0.repeat(1, batch_size, 1).to(self.device)
        else:
            weight = next(self.parameters()).data
            hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden
# ========================================================================================
class ActorCriticGRU(nn.Module):
    """
    사용자의 GRUPolicy 구조와 초기화 방식을 PPO에 맞게 수정한 액터-크리틱 네트워크.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=128, device='cuda', learn_h0=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.device = device

        # GRU 레이어 (사용자의 GRUPolicy와 동일한 구조)
        self.gru = nn.GRU(obs_dim, hidden_dim, 1, batch_first=True)
        
        # 정책(Actor)을 위한 출력 레이어
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # 가치(Critic)를 위한 출력 레이어
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        # 행동의 표준편차 (학습 가능한 파라미터)
        self.action_log_std = nn.Parameter(th.zeros(1, action_dim))

        # --- 사용자의 정교한 가중치 초기화 로직 적용 ---
        for name, param in self.named_parameters():
            if "gru.weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "gru.weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "gru.bias" in name:
                nn.init.zeros_(param)
            elif "actor_head.weight" in name:
                nn.init.xavier_uniform_(param)
            elif "actor_head.bias" in name:
                # Sigmoid가 아닌 Tanh를 사용하므로 bias를 0으로 초기화
                nn.init.zeros_(param)
            elif "critic_head.weight" in name:
                nn.init.xavier_uniform_(param)
            elif "critic_head.bias" in name:
                nn.init.zeros_(param)
        
        if learn_h0:
            self.h0 = nn.Parameter(th.zeros(self.n_layers, 1, hidden_dim), requires_grad=True)
    
    def forward(self, obs, h_prev):
        # [오류 수정] 입력 텐서의 형태(shape)를 GRU에 맞게 보정합니다.
        # GRU는 (batch, seq_len, features) 3D 텐서를 기대합니다.
        
        # 1. 단일 관측값 (select_action에서 호출): (features,) -> (1, 1, features)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        # 2. 배치 관측값 (train에서 호출): (batch, features) -> (batch, 1, features)
        elif obs.ndim == 2:
            obs = obs.unsqueeze(1)
        
        # 이제 obs는 항상 올바른 3D 형태를 가집니다.
        gru_out, h_next = self.gru(obs, h_prev)
        
        # 은닉 상태로부터 가치와 행동의 평균 계산
        value = self.critic_head(gru_out.squeeze(1))
        action_mean_raw = self.actor_head(gru_out.squeeze(1))
        # PPO는 보통 tanh를 사용해 행동 범위를 [-1, 1]로 제한
        action_mean = th.tanh(action_mean_raw)
        
        # 행동 분포 생성
        action_std = th.exp(self.action_log_std)
        dist = Normal(action_mean, action_std)
        
        return dist, value, h_next
    
    def init_hidden(self, batch_size):
        if hasattr(self, 'h0'):
            return self.h0.repeat(1, batch_size, 1).to(self.device)
        else:
            return th.zeros(self.n_layers, batch_size, self.hidden_dim, device=self.device)       
 
class GRUActor(nn.Module):
    """GRU를 사용하는 결정론적(Deterministic) 액터 네트워크 (for DDPG)"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128, device = 'cuda', learn_h0=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.device = device
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.net = nn.Linear(hidden_dim, action_dim)
           # --- 사용자의 정교한 가중치 초기화 로직 적용 ---
        for name, param in self.named_parameters():
            if "gru.weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "gru.weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "gru.bias" in name:
                nn.init.zeros_(param)
            elif "net.weight" in name:
                nn.init.xavier_uniform_(param)
            elif "net.bias" in name:
                nn.init.zeros_(param)
        if learn_h0:
            self.h0 = nn.Parameter(th.zeros(self.n_layers, 1, hidden_dim), requires_grad=True)
            
    def forward(self, obs, h_prev):
        # GRU는 (batch, seq_len, features) 3D 텐서를 기대합니다.
        
        # 1. 단일 관측값 (select_action에서 호출): (features,) -> (1, 1, features)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        # 2. 배치 관측값 (train에서 호출): (batch, features) -> (batch, 1, features)
        elif obs.ndim == 2:
            obs = obs.unsqueeze(1)
        
        gru_out, h_next = self.gru(obs, h_prev)
        action = self.net(gru_out.squeeze(1))
        action = th.sigmoid(action)  # DDPG는 보통 sigmoid를 사용해 행동 범위를 [0, 1]로 제한
        return action, h_next
    
    def init_hidden(self, batch_size):
        if hasattr(self, 'h0'):
            return self.h0.repeat(1, batch_size, 1).to(self.device)
        else:
            return th.zeros(self.n_layers, batch_size, self.hidden_dim, device=self.device)       
 
class GRUSACActor(nn.Module):
    """GRU를 사용하는 확률적(Stochastic) 액터 네트워크 (for SAC)"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, hidden_state):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        
        gru_out, h_next = self.gru(obs, hidden_state)
        gru_out = gru_out.squeeze(1)
        
        mean = self.mean_layer(gru_out)
        log_std = self.log_std_layer(gru_out)
        log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, h_next
    
    def sample(self, obs, hidden_state):
        mean, log_std, h_next = self.forward(obs, hidden_state)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        x_t = dist.rsample()  # Reparameterization Trick
        y_t = th.tanh(x_t)
        action = y_t
        
        log_prob = dist.log_prob(x_t)
        log_prob -= th.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob, h_next
    
    def init_hidden(self, batch_size):
        if hasattr(self, 'h0'):
            return self.h0.repeat(1, batch_size, 1).to(self.device)
        else:
            return th.zeros(self.n_layers, batch_size, self.hidden_dim, device=self.device)       
 
class GRUCritic(nn.Module):
    """GRU를 사용하는 크리틱 네트워크입니다."""
    def __init__(self, obs_dim, action_dim, hidden_dim=128, device='cuda', learn_h0=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = 1
        self.device = device

        self.gru = nn.GRU(obs_dim, hidden_dim, 1, batch_first=True)
        self.net = nn.Linear(hidden_dim + action_dim, 1)
          # --- 사용자의 정교한 가중치 초기화 로직 적용 ---
        for name, param in self.named_parameters():
            if "gru.weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "gru.weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "gru.bias" in name:
                nn.init.zeros_(param)
            elif "net.weight" in name:
                nn.init.xavier_uniform_(param)
            elif "net.bias" in name:
                nn.init.zeros_(param)
        if learn_h0:
            self.h0 = nn.Parameter(th.zeros(self.n_layers, 1, hidden_dim), requires_grad=True)
       

    def forward(self, obs, action, h_prev):
        # 1. 단일 관측값 (select_action에서 호출): (features,) -> (1, 1, features)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
        # 2. 배치 관측값 (train에서 호출): (batch, features) -> (batch, 1, features)
        elif obs.ndim == 2:
            obs = obs.unsqueeze(1)
        
            
        gru_out, _ = self.gru(obs, h_prev)        
        x = th.cat([gru_out.squeeze(1), action], dim=1)
        return self.net(x)
    
    def init_hidden(self, batch_size):
        if hasattr(self, 'h0'):
            return self.h0.repeat(1, batch_size, 1).to(self.device)
        else:
            return th.zeros(self.n_layers, batch_size, self.hidden_dim, device=self.device)       
 
    
       
# class GRUPolicyNet(nn.Module):
#     def __init__(self, obs_dim, hidden_dim, action_dim):
#         super().__init__()
#         self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)
#         self.fc_mean = nn.Linear(hidden_dim, action_dim)
#         # log_std as parameter
#         self.log_std = nn.Parameter(th.zeros(action_dim) - 1.0)
#     def forward(self, obs_seq, h0=None):
#         # obs_seq: (batch, seq_len, obs_dim) or (batch, obs_dim)
#         if obs_seq.ndim == 2:
#             x = obs_seq.unsqueeze(1)  # (batch,1,obs_dim)
#         else:
#             x = obs_seq
#         batch = x.size(0)
#         if h0 is None:
#             h0_ = x.new_zeros(1, batch, self.gru.hidden_size)
#         else:
#             h0_ = h0
#         y, h_n = self.gru(x, h0_)  # y: (batch, seq_len, hidden_dim)
#         out = y[:, -1, :]          # last output
#         mean = self.fc_mean(out)   # (batch, action_dim)
#         log_std = self.log_std.unsqueeze(0).expand(batch, -1)  # (batch, action_dim)
#         return mean, log_std, h_n

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims):
        super().__init__()
        # 입력: [obs, action]
        self.net = MLP(obs_dim + action_dim, hidden_dims, 1)
    def forward(self, obs, action):
        x = th.cat([obs, action], dim=-1)
        q = self.net(x)
        return q  # (batch,1)
