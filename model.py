import torch
import torch.nn as nn
import torch.nn.init as init
from torch.distributions import Categorical
from utils import reparameterization, gru_att


class InferenceNet(nn.Module):
    """
        Parameterizes the approximated distribution of h_t `q(h_t | h_{t-1}, s, x_{1:t})`
    """

    def __init__(self, h_dim, s_dim, hidden_dim_, gru_dim):
        super(InferenceNet, self).__init__()
        self.lin_h_to_hidden_ = nn.Sequential(nn.Linear(h_dim, hidden_dim_),
                                              nn.ReLU())
        self.lin_hidden_cmb_static = nn.Sequential(nn.Linear(hidden_dim_ + s_dim, gru_dim),
                                                   nn.ReLU())
        self.lin_hidden_to_mean = nn.Linear(gru_dim, h_dim)
        self.lin_hidden_to_std = nn.Linear(gru_dim, h_dim)
        init.kaiming_normal_(self.lin_h_to_hidden_[0].weight, mode='fan_in')
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

        proposed_hidden_ = self.lin_h_to_hidden_(h_t_1)
        proposed_hidden = self.lin_hidden_cmb_static(torch.cat((proposed_hidden_, s), dim=1))
        h_combined = 0.5 * self.tanh(proposed_hidden + h_gru)
        proposed_mean = self.lin_hidden_to_mean(h_combined)
        proposed_std = self.elu(self.lin_hidden_to_std(h_combined)) + 1 + 1e-12

        return proposed_mean, proposed_std


class Predictor(nn.Module):
    """
        Health Risk Prediction
    """
    def __init__(self, h_dim, drop_prob):
        super(Predictor, self).__init__()
        self.h_dim = h_dim

        self.visit = nn.Linear(h_dim, 1)
        self.w_hy = nn.Sequential(nn.Linear(h_dim, h_dim//2),
                                  nn.ReLU(),
                                  nn.Linear(h_dim//2, 2))
        init.kaiming_normal_(self.w_hy[0].weight, mode='fan_in')
        self.dropout = nn.Dropout(p=drop_prob)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h):
        """
            inputs :
                h : input hidden states ( B x T x h_dim )
            returns :
                output : the predicted class probability distribution ( B x num_class )
        """
        visit_att = self.softmax(self.visit(h))
        h_out = torch.bmm(visit_att.permute(0, 2, 1), h).squeeze(1)
        h_out = self.dropout(h_out)
        output = self.softmax(self.w_hy(h_out))

        return output


class RiskPre(nn.Module):
    """
        The Proposed Model
    """
    def __init__(self, input_dim, h_dim, s_dim, gru_dim, gru_num_layers, gru_dropout_rate, hidden_dim_, drop_prob, device):
        super(RiskPre, self).__init__()
        self.h_dim = h_dim
        self.device = device
        self.h_q_0 = nn.Parameter(torch.zeros(h_dim), requires_grad=True).to(device)
        self.mu_q_0 = nn.Parameter(torch.zeros(h_dim), requires_grad=True).to(device)
        self.var_q_0 = nn.Parameter(torch.zeros(h_dim), requires_grad=True).to(device)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, gru_dim), requires_grad=True).to(device)

        gru_dropout_rate = 0. if gru_num_layers == 1 else gru_dropout_rate
        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_dim, batch_first=True,
                          bidirectional=False, num_layers=gru_num_layers, dropout=gru_dropout_rate)
        self.InferNet = InferenceNet(h_dim, s_dim, hidden_dim_, gru_dim)
        self.predictor = Predictor(h_dim, drop_prob)

    def forward(self, x, s):
        """
            input :
                x : time-variant features ( B x T x input_dim )
                s : time-invariant features ( B x s_dim )
            returns :
                output : the predicted class probability distribution ( B x num_class )
                h_gru : a series of hidden states obtained from GRU ( B x T x gru_dim )
                mu_q_seq : the mean of the approximated distribution ( B x T x h_dim )
                var_q_seq :  the variance of the approximated distribution ( B x T x h_dim )
        """
        batch_size = x.size(0)
        t_max = x.size(1)

        # 1.GRU
        h_0_contig = self.h_0.expand(1, batch_size, self.gru.hidden_size).contiguous()
        h_gru, _ = self.gru(x, h_0_contig)

        # 2.Inference Network
        h_q_0 = self.h_q_0.expand(batch_size, self.h_dim)
        mu_q_0 = self.mu_q_0.expand(batch_size, self.h_dim)
        var_q_0 = self.var_q_0.expand(batch_size, self.h_dim)
        mu_q_seq = torch.zeros([batch_size, t_max, self.h_dim]).to(self.device)
        var_q_seq = torch.zeros([batch_size, t_max, self.h_dim]).to(self.device)

        for t in range(0, t_max):
            if t == 0:
                mu_q_seq[:, t, :] = mu_q_0
                var_q_seq[:, t, :] = var_q_0
                h_prev_all_q = h_q_0.unsqueeze(1)
                continue

            mu_q, var_q = self.InferNet(h_prev_all_q[:, t - 1, :], s, h_gru[:, t, :])
            ht_q = reparameterization(mu_q, var_q)
            h_prev_all_q = torch.cat((h_prev_all_q, ht_q.unsqueeze(1)), 1)
            mu_q_seq[:, t, :] = mu_q
            var_q_seq[:, t, :] = var_q

        # 3.Predictor
        output = self.predictor(h_prev_all_q)

        return output, h_gru, mu_q_seq, var_q_seq


class Transition(nn.Module):
    """
        Parameterizes the state transition distribution `p(h_t | h_i ,s)`
    """
    def __init__(self, h_dim, s_dim, transition_dim_, transition_dim):
        super(Transition, self).__init__()
        self.transition_dim_ = transition_dim_
        self.lin_hidden = nn.Sequential(nn.Linear(h_dim, transition_dim_),
                                        nn.ReLU())
        self.lin_hidden_cmb_static = nn.Sequential(nn.Linear(transition_dim_ + s_dim, transition_dim),
                                                   nn.ReLU())
        self.lin_hidden_to_mean = nn.Linear(transition_dim, h_dim)
        self.lin_hidden_to_std = nn.Linear(transition_dim, h_dim)
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
            proposed_std = self.elu(self.lin_hidden_to_std(proposed_hidden)) + 1 + 1e-12
            proposed_mean_list.append(proposed_mean)
            proposed_std_list.append(proposed_std)

        mean = torch.stack(proposed_mean_list)
        std = torch.stack(proposed_std_list)
        return mean, std


class Emission(nn.Module):
    """
        Parameterizes the emission distribution `p(x_t | h_t)`
    """

    def __init__(self, h_dim, emission_dim, x_dim):
        super(Emission, self).__init__()
        self.lin_z_to_hidden = nn.Sequential(nn.Linear(h_dim, emission_dim),
                                             nn.ReLU())
        self.lin_hidden_to_mean = nn.Linear(emission_dim, x_dim)
        self.lin_hidden_to_std = nn.Linear(emission_dim, x_dim)
        self.elu = nn.ELU()

    def forward(self, h_t):
        """
            inputs :
                h_t : input hidden states at time t ( B x h_dim )
            returns :
                xt_recon : Obtained by sampling from the emission distribution (B x num_class )
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
        Transparency and interpretability
    """
    def __init__(self, h_dim, s_dim, emission_dim, x_dim, transition_dim_, transition_dim, device):
        super(NeuralHmm, self).__init__()
        self.h_dim = h_dim
        self.x_dim = x_dim
        self.device = device

        self.h_p_0 = nn.Parameter(torch.zeros(h_dim), requires_grad=True).to(device)
        self.mu_p_0 = nn.Parameter(torch.zeros(h_dim), requires_grad=True).to(device)
        self.var_p_0 = nn.Parameter(torch.zeros(h_dim), requires_grad=True).to(device)

        self.transition = Transition(h_dim, s_dim, transition_dim_, transition_dim)
        self.emission = Emission(h_dim, emission_dim, x_dim)

    def forward(self, x, s, h_gru):
        """
            input :
                x : time-variant features ( B x T x input_dim )
                s : time-invariant features ( B x s_dim )
                h_gru : a series of hidden states obtained from GRU ( B x T x gru_dim )
            returns :
                x_recon : Obtained by sampling from the emission distribution (B x num_class )
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

        x_recon = torch.zeros([batch_size, t_max, self.x_dim]).to(self.device)
        mu_p_seq = torch.zeros([batch_size, t_max, self.h_dim]).to(self.device)
        var_p_seq = torch.zeros([batch_size, t_max, self.h_dim]).to(self.device)

        # 2.Neural hmm
        for t in range(0, t_max):
            if t == 0:
                x_recon[:, t, :] = x[:, 0, :]
                mu_p_seq[:, t, :] = mu_p_0.squeeze(1)
                var_p_seq[:, t, :] = var_p_0.squeeze(1)
                h_prev_all_p = h_p_0.unsqueeze(1)
                continue

            alpha = gru_att(h_gru, t)
            categorical_alpha = Categorical(alpha)
            pis_alpha = categorical_alpha.sample()

            mu_p, var_p = self.transition(h_prev_all_p[:, 0:t, :], s)
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
