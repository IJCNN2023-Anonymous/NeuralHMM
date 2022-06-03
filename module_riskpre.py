import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Categorical
from utils import reparameterization


class InferenceNet(nn.Module):
    """
        Parameterizes the approximated distribution of h_t `q(h_t | h_{t-1}, s, x_{1:t})`
    """

    def __init__(self, h_dim, static_dim, hidden_dim_, gru_dim):
        super(InferenceNet, self).__init__()
        self.lin_z_to_hidden_ = nn.Sequential(nn.Linear(h_dim, hidden_dim_),
                                              nn.ReLU(),
                                              nn.BatchNorm1d(hidden_dim_))
        self.lin_hidden_cmb_static = nn.Sequential(nn.Linear(hidden_dim_ + static_dim, gru_dim),
                                                   nn.ReLU(),
                                                   nn.BatchNorm1d(gru_dim))
        self.lin_hidden_to_mean = nn.Linear(gru_dim, h_dim)
        self.lin_hidden_to_std = nn.Linear(gru_dim, h_dim)
        init.kaiming_normal_(self.lin_z_to_hidden_[0].weight, mode='fan_in')
        init.kaiming_normal_(self.lin_hidden_cmb_static[0].weight, mode='fan_in')
        init.xavier_normal_(self.lin_hidden_to_mean.weight)
        init.xavier_normal_(self.lin_hidden_to_std.weight)
        self.tanh = nn.Tanh()
        self.elu = nn.ELU()

    def forward(self, h_t_1, s, h_gru):
        """
            inputs :
                h_t_1 : input hidden states at time t-1 ( B x h_dim )
                s : time-invariant features ( B x s_dim )
                h_gru : a series of hidden states obtained from GRU ( B x T x gru_dim )
            returns :
                proposed_mean : the mean of the approximated distribution ( B x h_dim )
                proposed_std :  the variance of the approximated distribution ( B x h_dim )
        """
        h_t_1 = h_t_1.float()
        s = s.float()
        h_gru = h_gru.float()

        proposed_hidden_ = self.lin_z_to_hidden_(h_t_1)
        proposed_hidden = self.lin_hidden_cmb_static(torch.cat((proposed_hidden_, s), dim=1))
        h_combined = 0.5 * self.tanh(proposed_hidden + h_gru)
        proposed_mean = self.lin_hidden_to_mean(h_combined)
        proposed_std = self.elu(self.lin_hidden_to_std(h_combined)) + 1 + 1e-12

        return proposed_mean, proposed_std


class MdnPredictor(nn.Module):
    """
        作用于health risk prediction的预测器
    """
    def __init__(self, h_dim, s_dim, seq_len, out_dim, n_gaussian):
        super(MdnPredictor, self).__init__()
        self.h_dim = h_dim
        self.n_gaussian = n_gaussian
        self.num_class = 2

        self.hidden_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h_dim * seq_len, out_dim),
            nn.ReLU()
        )

        self.pi_proj = nn.Linear(out_dim + s_dim, n_gaussian)
        self.mu_proj = nn.Linear(out_dim + s_dim, self.num_class * n_gaussian)
        self.sigma_proj = nn.Linear(out_dim + s_dim, self.num_class * n_gaussian)

        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, s):
        """
            inputs :
                h : input hidden states ( B x T x h_dim )
                s : time-invariant features ( B x s_dim )
            returns :
                sample_c : the predicted class probability distribution ( B x num_class )
        """
        hidden_e = self.hidden_proj(h)
        hidden_e = torch.cat((hidden_e, s), 1)

        pi = self.softmax(self.pi_proj(hidden_e))
        mu = self.mu_proj(hidden_e)
        sigma = self.elu(self.sigma_proj(hidden_e)) + 1 + 1e-12

        mu = mu.view(-1, self.num_class, self.n_gaussians).permute(2, 0, 1)
        sigma = sigma.view(-1, self.num_class, self.n_gaussians).permute(2, 0, 1)

        categorical_pi = Categorical(pi)
        pis = categorical_pi.sample()

        mu_ = torch.gather(mu, 0, pis.unsqueeze(1).expand(mu.shape[1], mu.shape[2]).unsqueeze(0))
        sigma_ = torch.gather(sigma, 0, pis.unsqueeze(1).expand(sigma.shape[1], sigma.shape[2]).unsqueeze(0))
        mu_ = mu_.squeeze(0)
        sigma_ = sigma_.squeeze(0)
        sample_c = self.softmax(reparameterization(mu_, sigma_))

        return sample_c


class RiskPre(nn.Module):
    """
        整个网络for performing health risk prediction tasks
    """
    def __init__(self, input_dim, h_dim, s_dim, gru_dim, gru_num_layers, gru_dropout_rate, hidden_dim_, out_dim, n_gaussian, device):
        super(RiskPre, self).__init__()
        self.h_dim = h_dim
        self.h_q_0 = nn.Parameter(torch.zeros(h_dim), requires_grad=True).to(device)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, gru_dim), requires_grad=True).to(device)

        gru_dropout_rate = 0. if gru_num_layers == 1 else gru_dropout_rate
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_dim, batch_first=True,
                          bidirectional=False, num_layers=gru_num_layers, dropout=gru_dropout_rate)
        self.InferNet = InferenceNet(h_dim, s_dim, hidden_dim_, gru_dim)
        self.predictor = MdnPredictor(h_dim, out_dim, n_gaussian=n_gaussian)

    def forward(self, x, s):
        """
            input :
                x : time-variant features ( B x T x input_dim )
                s : time-invariant features ( B x s_dim )
            returns :
                sample_c : the predicted class probability distribution ( B x num_class )
        """
        batch_size = x.size(0)
        t_max = x.size(1)

        # 1.GRU
        h_0_contig = self.h_0.expand(1, batch_size, self.gru.hidden_size).contiguous()
        h_gru, _ = self.gru(x, h_0_contig)

        # 2.Inference Network
        h_q_0 = self.h_q_0.expand(batch_size, self.h_dim)
        for t in range(0, t_max):
            # Z0是初始化生成的，hmm直接从求解Z1开始
            if t == 0:
                h_prev_all_q = h_q_0.unsqueeze(1)
                continue

            mu_q, var_q = self.InferNet(h_prev_all_q[:, t - 1, :], s, h_gru[:, t, :])
            ht_q = reparameterization(mu_q, var_q)
            h_prev_all_q = torch.cat((h_prev_all_q, ht_q.unsqueeze(1)), 1)

        # 3.MDN Predictor
        sample_c = self.predictor(h_prev_all_q, s)

        return sample_c, h_gru, mu_q, var_q, h_prev_all_q

