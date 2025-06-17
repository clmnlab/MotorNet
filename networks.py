import torch as th
import torch.nn as nn

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

import torch as th


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
