import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import class_weight
from model import RiskPre, NeuralHmm
from utils import print_metrics_binary, setup_seed, neuralhmm_loss, data_extract
from data_process import data_process_x, data_process_s


def train_riskpre(train_loader,
                  valid_loader,
                  idx_list,
                  demographic_data,
                  diagnosis_data,
                  file_name,
                  input_dim,
                  h_dim,
                  s_dim,
                  gru_dim,
                  gru_num_layers,
                  gru_dropout_rate,
                  hidden_dim_,
                  seq_len,
                  out_dim,
                  n_gaussian,
                  lr,
                  epochs,
                  seed,
                  device,
                  model_name = 'RiskPre'):

    model = RiskPre(input_dim, h_dim, s_dim, gru_dim, gru_num_layers, gru_dropout_rate, hidden_dim_, seq_len, out_dim, n_gaussian, device).to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)

    setup_seed(seed)
    train_loss = []
    valid_loss = []
    best_epoch = 0
    best_auroc = 0
    best_auprc = 0
    history = []
    train_out = []
    valid_out = []

    for each_epoch in range(epochs):
        batch_loss = []
        model.train()

        h_gru_train = []
        h_gru_valid = []
        mu_q_train = []
        mu_q_valid = []
        var_q_train = []
        var_q_valid = []
        batch_x_train = []
        batch_x_valid = []
        batch_y_train = []
        batch_y_valid = []
        batch_name_train = []
        batch_name_valid = []

        for step, (batch_x, batch_y, batch_name) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_name_train.append(batch_name)
            batch_x_train.append(batch_x.cpu().detach().numpy())
            batch_y_train.append(batch_y.cpu().detach().numpy())

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
            sample_c, h_gru, mu_q, var_q = model(batch_x, batch_s)
            h_gru_train.append(h_gru.cpu().detach().numpy())
            mu_q_train.append(mu_q.cpu().detach().numpy())
            var_q_train.append(var_q.cpu().detach().numpy())

            batch_y = batch_y.long()
            y_out = batch_y.cpu().numpy()
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_out), y=y_out)
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

            loss = criterion(sample_c, batch_y)
            batch_loss.append(loss.cpu().detach().numpy())

            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

            if step % 10 == 0:
                print('Epoch %d Batch %d: Train Loss = %.4f' % (each_epoch, step, np.mean(np.array(batch_loss))))

        train_loss.append(np.mean(np.array(batch_loss)))
        batch_loss = []

        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()

            for step, (batch_x, batch_y, batch_name) in enumerate(valid_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_name_valid.append(batch_name)
                batch_x_valid.append(batch_x.cpu().detach().numpy())
                batch_y_valid.append(batch_y.cpu().detach().numpy())

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
                sample_c, h_gru, mu_q, var_q = model(batch_x, batch_s)
                h_gru_valid.append(h_gru.cpu().detach().numpy())
                mu_q_valid.append(mu_q.cpu().detach().numpy())
                var_q_valid.append(var_q.cpu().detach().numpy())

                batch_y = batch_y.long()
                y_out = batch_y.cpu().numpy()
                class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_out), y=y_out)
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

                loss = criterion(sample_c, batch_y)
                batch_loss.append(loss.cpu().detach().numpy())

                y_pred.append(sample_c)
                y_true.append(batch_y)

        valid_loss.append(np.mean(np.array(batch_loss)))
        print('\n==>Predicting on validation')
        print('Valid Loss = %.4f' % (valid_loss[-1]))
        print()
        y_pred = torch.cat(y_pred, 0)
        y_true = torch.cat(y_true, 0)
        test_y_pred = y_pred.cpu().detach().numpy()
        test_y_true = y_true.cpu().detach().numpy()
        ret = print_metrics_binary(test_y_true, test_y_pred)
        history.append(ret)

        cur_auroc = ret['auroc']

        if cur_auroc > best_auroc:
            best_epoch = each_epoch
            best_auroc = ret['auroc']
            best_auprc = ret['auprc']
            state = {
                'net': model.state_dict(),
                'optimizer': opt_model.state_dict(),
                'epoch': each_epoch
            }
            torch.save(state, file_name)

        train_out.extend([[h_gru_train, mu_q_train, var_q_train, batch_x_train, batch_y_train, batch_name_train]])
        valid_out.extend([[h_gru_valid, mu_q_valid, var_q_valid, batch_x_valid, batch_y_valid, batch_name_valid]])

    return best_epoch, best_auroc, best_auprc, history, train_out, valid_out


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
    min_loss = 99999

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
        print()

        if valid_loss_ < min_loss:
            min_loss = valid_loss_
            state = {
                'net': model.state_dict(),
                'optimizer': opt_model.state_dict(),
                'epoch': each_epoch
            }
            torch.save(state, file_name)

    return min_loss

if __name__ == '__main__':
    # parameters
    data_x_path = './data_x/'
    data_s_path = './data_s/'
    file_name_riskpre = './model/riskpre'
    input_dim = 76
    h_dim = 12
    s_dim = 132
    gru_dim = 36
    gru_num_layers = 1
    gru_dropout_rate = 0
    hidden_dim_ = 16
    seq_len = 60
    out_dim = 17
    n_gaussian = 8
    lr = 1e-3
    epochs = 100
    seed = 2022
    device = 'cuda:0'

    train_loader, valid_loader, _ = data_process_x(data_x_path)
    demographic_data, diagnosis_data, idx_list = data_process_s(data_s_path)

    best_epoch, best_auroc, best_auprc, history, best_epoch_train, best_epoch_valid = train_riskpre(train_loader, valid_loader, idx_list, \
                  demographic_data, diagnosis_data, file_name_riskpre, input_dim, h_dim, s_dim, gru_dim, gru_num_layers, \
                  gru_dropout_rate, hidden_dim_, seq_len, out_dim, n_gaussian, lr, epochs, seed, device, model_name = 'RiskPre')
    print('-------------best performance----------')
    print('Best Auroc = %.4f ' %(best_auroc))
    print('Best Auprc = %.4f ' %(best_auprc))
    print()

    # parameters
    file_name_neuralhmm = './model/neuralhmm'
    emission_dim = 32
    x_dim = 76
    transition_dim_ = 32
    transition_dim = 24
    lr = 1e-3
    epochs = 100
    seed = 2022
    device = 'cuda:0'

    h_gru_train, mu_q_train, var_q_train, batch_x_train, batch_y_train, batch_name_train = data_extract(best_epoch_train, best_epoch)
    h_gru_valid, mu_q_valid, var_q_valid, batch_x_valid, batch_y_valid, batch_name_valid = data_extract(best_epoch_valid, best_epoch)
    min_loss = train_neuralhmm(idx_list, demographic_data, diagnosis_data, h_gru_train, h_gru_valid, mu_q_train, mu_q_valid, \
                    var_q_train, var_q_valid, batch_x_train, batch_x_valid, batch_name_train, batch_name_valid, file_name_neuralhmm, \
                    h_dim, s_dim, emission_dim, x_dim, transition_dim_, transition_dim, lr, epochs, seed, device, model_name='NeuralHmm')
    print('-------------best performance----------')
    print('Neural Hmm Loss = %.4f ' % (min_loss))
