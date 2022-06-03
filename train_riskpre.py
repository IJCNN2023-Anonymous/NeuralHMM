import random
import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import class_weight
from module_riskpre import RiskPre
from utils import print_metrics_binary


def train_riskpre(train_loader,
                  valid_loader,
                  idx_list,
                  demographic_data,
                  diagnosis_data,
                  input_dim,
                  h_dim,
                  s_dim,
                  gru_dim,
                  gru_num_layers,
                  gru_dropout_rate,
                  hidden_dim_,
                  out_dim,
                  n_gaussian,
                  lr,
                  epochs,
                  seed,
                  device,
                  file_name,
                  model_name = 'RiskPre'):
    model = RiskPre(input_dim, h_dim, s_dim, gru_dim, gru_num_layers, gru_dropout_rate, hidden_dim_, out_dim, n_gaussian, device).to(device)
    opt_model = torch.optim.Adam(model.parameters(), lr=lr)

    setup_seed(seed)
    train_loss = []
    valid_loss = []
    best_auroc = 0
    best_auprc = 0
    history = []
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    best_epoch_train = []
    best_epoch_valid = []

    for each_epoch in range(epochs):
        batch_loss = []
        model.train()

        h_prev_all_q_train = []
        h_prev_all_q_valid = []
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
            sample_c, h_gru, mu_q, var_q, h_prev_all_q = model(batch_x, torch.cat((batch_demo, batch_diag),1))
            h_gru_train.append(h_gru.cpu().detach().numpy())
            h_prev_all_q_train.append(h_prev_all_q.cpu().detach().numpy())
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
                sample_c, h_gru, mu_q, var_q, h_prev_all_q = model(batch_x, torch.cat((batch_demo, batch_diag), 1))
                h_gru_valid.append(h_gru.cpu().detach().numpy())
                h_prev_all_q_valid.append(h_prev_all_q.cpu().detach().numpy())
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
        y_pred = torch.cat(y_pred, 0)
        y_true = torch.cat(y_true, 0)
        test_y_pred = y_pred.cpu().detach().numpy()
        test_y_true = y_true.cpu().detach().numpy()
        ret = print_metrics_binary(test_y_true, test_y_pred)
        history.append(ret)
        print()

        cur_auroc = ret['auroc']

        if cur_auroc > best_auroc:
            best_auroc = ret['auroc']
            best_auprc = ret['auprc']
            print('-------------best performance----------')
            print(best_auroc)
            print(best_auprc)
            state = {
                'net': model.state_dict(),
                'optimizer': opt_model.state_dict(),
                'epoch': each_epoch
            }
            torch.save(state, file_name)
            best_epoch_train.extend([h_prev_all_q_train, h_gru_train, mu_q_train, var_q_train, batch_x_train, batch_y_train, batch_name_train])
            best_epoch_valid.extend([h_prev_all_q_valid, h_gru_valid, mu_q_valid, var_q_valid, batch_x_valid, batch_y_valid, batch_name_valid])
            print('------------ Save best model ------------')

    return history, best_epoch_train, best_epoch_valid


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
