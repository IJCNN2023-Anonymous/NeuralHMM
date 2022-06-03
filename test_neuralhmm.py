import torch
import numpy as np
from module_neuralhmm import NeuralHmm
from utils import setup_seed, neuralhmm_loss


def test_riskpre(idx_list,
                 demographic_data,
                 diagnosis_data,
                 h_gru_test,
                 mu_q_test,
                 var_q_test,
                 batch_x_test,
                 batch_name_test,
                 file_name,
                 h_dim=5,
                 s_dim=132,
                 emission_dim=32,
                 out_dim=17,
                 transition_dim_=32,
                 transition_dim=24,
                 seed=2022,
                 device='cuda:0',
                 model_name='NeuralHmm'):

    setup_seed(seed)
    model = NeuralHmm(h_dim, s_dim, emission_dim, out_dim, transition_dim_, transition_dim, device).to(device)
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['net'])
    model.eval()
    batch_loss = []

    for step in range(len(batch_x_test)):
        batch_x = torch.tensor(batch_x_test[step]).to(device)
        batch_name = batch_name_test[step]

        batch_demo = []
        batch_diag = []
        for i in range(len(batch_name)):
            cur_id, cur_ep, _ = batch_name[i].split('_', 2)
            cur_idx = cur_id + '_' + cur_ep
            idx = idx_list.index(cur_idx) if cur_idx in idx_list else None
            if idx == None:
                cur_demo = torch.zeros(4)
                cur_diag = torch.zeros(128)
            else:
                cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)
                cur_diag = torch.tensor(diagnosis_data[idx], dtype=torch.float32)
            batch_demo.append(cur_demo)
            batch_diag.append(cur_diag)

        batch_demo = torch.stack(batch_demo).to(device)
        batch_diag = torch.stack(batch_diag).to(device)
        batch_s = torch.cat((batch_demo, batch_diag), 1)
        h_gru = torch.tensor(h_gru_test[step]).to(device)
        mu_q_seq = torch.tensor(mu_q_test[step]).to(device)
        var_q_seq = torch.tensor(var_q_test[step]).to(device)

        x_recon, x, mu_p_seq, var_p_seq = model(batch_x, batch_s, h_gru)

        kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, loss = \
            neuralhmm_loss(x, x_recon, mu_q_seq, var_q_seq, mu_p_seq, var_p_seq)
        batch_loss.append(loss.cpu().detach().numpy())

    test_loss_ = np.mean(np.array(batch_loss))
    print('\n==>Predicting on test')
    print('Test Loss = %.4f' % test_loss_)

    return test_loss_
