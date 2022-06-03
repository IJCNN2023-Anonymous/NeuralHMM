import torch
import numpy as np
from module_neuralhmm import NeuralHmm
from utils import setup_seed, neuralhmm_loss


def train_neuralhmm(idx_list,
                  demographic_data,
                  diagnosis_data,
                  h_gru_train,
                  h_gru_valid,
                  mu_q_train,
                  mu_q_valid,
                  var_q_train,
                  var_q_valid,
                  batch_x_train,
                  batch_x_valid,
                  batch_name_train,
                  batch_name_valid,
                  file_name,
                  h_dim = 5,
                  s_dim = 132,
                  emission_dim = 32,
                  out_dim = 17,
                  transition_dim_ = 32,
                  transition_dim = 24,
                  lr = 1e-3,
                  epochs = 100,
                  seed = 2022,
                  device = 'cuda:0',
                  model_name = 'NeuralHmm'):

    model = NeuralHmm(h_dim, s_dim, emission_dim, out_dim, transition_dim_, transition_dim, device).to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)

    setup_seed(seed)
    train_loss = []
    valid_loss = []
    max_loss = 99999

    for each_epoch in range(epochs):
        batch_loss = []
        model.train()

        for step in range(len(batch_x_train)):
            batch_x = torch.tensor(batch_x_train[step]).to(device)
            batch_name = batch_name_train[step]

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
            h_gru = torch.tensor(h_gru_train[step]).to(device)
            mu_q_seq = torch.tensor(mu_q_train[step]).to(device)
            var_q_seq = torch.tensor(var_q_train[step]).to(device)

            x_recon, x, mu_p_seq, var_p_seq = model(batch_x, batch_s, h_gru)

            kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, loss = \
                neuralhmm_loss(x, x_recon, mu_q_seq, var_q_seq, mu_p_seq, var_p_seq)
            batch_loss.append(loss.cpu().detach().numpy())

            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

            if step % 10 == 0:
                print('Epoch %d Batch %d: Train Loss = %.4f ' %(each_epoch, step, np.mean(np.array(batch_loss))))

        train_loss.append(np.mean(np.array(batch_loss)))

        batch_loss = []
        with torch.no_grad():
            model.eval()

            for step in range(len(batch_x_valid)):
                batch_x = torch.tensor(batch_x_valid[step]).to(device)
                batch_name = batch_name_valid[step]

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
                h_gru = torch.tensor(h_gru_valid[step]).to(device)
                mu_q_seq = torch.tensor(mu_q_valid[step]).to(device)
                var_q_seq = torch.tensor(var_q_valid[step]).to(device)

                x_recon, x, mu_p_seq, var_p_seq = model(batch_x, batch_s, h_gru)

                kl_raw, nll_raw, kl_fr, nll_fr, kl_m, nll_m, loss = \
                    neuralhmm_loss(x, x_recon, mu_q_seq, var_q_seq, mu_p_seq, var_p_seq)
                batch_loss.append(loss.cpu().detach().numpy())

        valid_loss_ = np.mean(np.array(batch_loss))
        valid_loss.append(valid_loss_)
        print('\n==>Predicting on validation')
        print('Valid Loss = %.4f' % valid_loss_)

        if valid_loss_ < max_loss:
            state = {
                'net': model.state_dict(),
                'optimizer': opt_model.state_dict(),
                'epoch': each_epoch
            }
            torch.save(state, file_name)
            print('------------ Save best model ------------')
    return
