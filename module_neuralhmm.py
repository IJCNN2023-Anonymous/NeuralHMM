import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Categorical
from utils import reparameterization, gru_att


class Transition(nn.Module):
    """
        Parameterizes the state transition distribution `p(h_t | h_i ,s)`
    """
    def __init__(self, h_dim, static_dim, transition_dim_, transition_dim):
        super(Transition, self).__init__()
        self.transition_dim_ = transition_dim_
        self.lin_hidden = nn.Sequential(nn.Linear(h_dim, transition_dim_),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(transition_dim_))
        self.lin_hidden_cmb_static = nn.Sequential(nn.Linear(transition_dim_ + static_dim, transition_dim),
                                                   nn.ReLU(),
                                                   nn.BatchNorm1d(transition_dim))
        self.lin_hidden_to_mean = nn.Linear(transition_dim, h_dim)
        self.lin_hidden_to_std = nn.Linear(transition_dim, h_dim)
        self.norm_std = nn.BatchNorm1d(h_dim)
        init.kaiming_normal_(self.lin_hidden[0].weight, mode='fan_in')
        init.kaiming_normal_(self.lin_hidden_cmb_static[0].weight, mode='fan_in')
        init.xavier_normal_(self.lin_hidden_to_mean.weight)
        self.elu = nn.ELU()

    def forward(self, h, s):
        """
            inputs :
                h : input hidden states ( B x T x h_dim )
                s : time-invariant features ( B x s_dim )
            returns :
                mean : the mean of the state transition distribution ( B x T x h_dim )
                std :  the variance of the state transition distribution ( B x T x h_dim )
        """
        h = h.float()
        s = s.float()
        proposed_mean_list = []
        proposed_std_list = []
        for t in range(h.shape[1]):
            proposed_hidden_ = self.lin_hidden(h[:, t, :])
            proposed_hidden = self.lin_hidden_cmb_static(torch.cat((proposed_hidden_, s), dim=1))
            proposed_mean = self.lin_hidden_to_mean(proposed_hidden)
            proposed_std = self.norm_std(self.elu(self.lin_hidden_to_std(proposed_hidden)) + 1 + 1e-12)
            proposed_mean_list.append(proposed_mean)
            proposed_std_list.append(proposed_std)

        mean = torch.stack(proposed_mean_list)
        std = torch.stack(proposed_std_list)
        return mean, std


class Emission(nn.Module):
    """
        Parameterizes the emission distribution `p(x_t | h_t)`
    """

    def __init__(self, h_dim, emission_dim, out_dim):
        super(Emission, self).__init__()
        self.lin_z_to_hidden = nn.Sequential(nn.Linear(h_dim, emission_dim),
                                             nn.ReLU(),
                                             nn.BatchNorm1d(emission_dim))
        self.lin_hidden_to_mean = nn.Linear(emission_dim, out_dim)
        self.lin_hidden_to_std = nn.Linear(emission_dim, out_dim)
        init.kaiming_normal_(self.lin_z_to_hidden[0].weight, mode='fan_in')
        init.xavier_normal_(self.lin_hidden_to_mean.weight)
        self.elu = nn.ELU()

    def forward(self, h_t):
        """
            inputs :
                h_t : input hidden states at time t ( B x h_dim )
            returns :
                xt_recon : 从emission distribution中采样得到的表示
                proposed_mean : the mean of the emission distribution ( B x out_dim )
                proposed_std :  the variance of the emission distribution ( B x out_dim )
        """
        proposed_hidden = self.lin_z_to_hidden(h_t)
        proposed_mean = self.lin_hidden_to_mean(proposed_hidden)
        proposed_std = self.elu(self.lin_hidden_to_std(proposed_hidden)) + 1 + 1e-12
        xt_recon = reparameterization(proposed_mean, proposed_std)

        return xt_recon, proposed_mean, proposed_std


class NeuralHmm(nn.Module):
    """
        一个post-hoc可解释模型，用于对GRU的隐藏状态进行解释
    """
    def __init__(self, h_dim, s_dim, emission_dim, out_dim, transition_dim_, transition_dim, device):
        super(NeuralHmm, self).__init__()
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.device = device

        self.z_q_0 = nn.Parameter(torch.zeros(h_dim), requires_grad=True).to(device)
        self.mu_p_0 = nn.Parameter(torch.zeros(h_dim), requires_grad=True).to(device)
        self.var_p_0 = nn.Parameter(torch.zeros(h_dim), requires_grad=True).to(device)

        self.transition = Transition(h_dim, s_dim, transition_dim_, transition_dim)
        self.emission = Emission(h_dim, emission_dim, out_dim)

    def forward(self, x, s, h_gru):
        """
            input :
                x : time-variant features ( B x T x input_dim )
                s : time-invariant features ( B x s_dim )
                h_gru : a series of hidden states obtained from GRU ( B x T x gru_dim )
            returns :
                x_recon : 从emission distribution中采样得到的表示 ( B x num_class )
                x : time-variant features ( B x T x input_dim )
                mu_p_seq : the mean of the state transition distribution ( B x T x h_dim )
                var_p_seq : the variance of the state transition distribution ( B x T x h_dim )
        """
        batch_size = x.size(0)
        t_max = x.size(1)

        # 1.Initial
        h_p_0 = self.h_p_0.expand(batch_size, self.h_dim)
        mu_p_0 = self.mu_p_0.expand(batch_size, 1, self.h_dim)
        var_p_0 = self.var_p_0.expand(batch_size, 1, self.h_dim)

        x_recon = torch.zeros([batch_size, t_max, self.out_dim]).to(self.device)
        mu_p_seq = torch.zeros([batch_size, t_max, self.h_dim]).to(self.device)
        var_p_seq = torch.zeros([batch_size, t_max, self.h_dim]).to(self.device)

        # 2.Neural hmm
        for t in range(0, t_max):
            # Z0是初始化生成的，hmm直接从求解Z1开始
            if t == 0:
                x_recon[:, t, :] = x[:, 0, :]
                mu_p_seq[:, t, :] = mu_p_0.squeeze(1)
                var_p_seq[:, t, :] = var_p_0.squeeze(1)
                h_prev_all_p = h_p_0.unsqueeze(1)
                continue

            alpha = gru_att(h_gru, t)
            categorical_alpha = Categorical(alpha)
            pis_alpha = categorical_alpha.sample()

            mu_p, var_p = self.transition(h_prev_all_p[:, 0:t, :], s)  # seq_len × batch_size × h_dim
            mu_ = torch.gather(mu_p, 0, pis_alpha.unsqueeze(1).expand(mu_p.shape[1], mu_p.shape[2]).unsqueeze(0))
            var_ = torch.gather(var_p, 0, pis_alpha.unsqueeze(1).expand(var_p.shape[1], var_p.shape[2]).unsqueeze(0))
            mu_ = mu_.squeeze(0)
            var_ = var_.squeeze(0)
            ht_p = reparameterization(mu_, var_)
            h_prev_all_p = torch.cat((h_prev_all_p, ht_p.unsqueeze(1)), 1)

            xt_recon, proposed_mean, proposed_std = self.emission(ht_p)
            x_recon[:, t, :] = xt_recon
            mu_p_seq[:, t, :] = mu_p[t - 1, :, :]
            var_p_seq[:, t, :] = var_p[t - 1, :, :]

        return x_recon, x, mu_p_seq, var_p_seq
