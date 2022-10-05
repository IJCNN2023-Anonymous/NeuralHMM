import torch
import pandas as pd
import numpy as np
import math
import time
import random
from torch.distributions import Normal
from sparsemax import Sparsemax
from sklearn import metrics


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reparameterization(mu, var):
    eps = torch.randn_like(var)
    return mu + eps * torch.sqrt(var)


def kl_div(mu1, var1, mu2=None, var2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if var2 is None:
        var2 = torch.zeros_like(mu1)

    return 0.5 * (
        var2 - var1 + (
            torch.exp(var1) + (mu1 - mu2).pow(2)
        ) / torch.exp(var2) - 1)


def nll_loss(x_hat, x):
    assert x_hat.dim() == x.dim() == 3
    assert x.size() == x_hat.size()
    return torch.nn.MSELoss(reduction='mean')(x_hat, x)


def determine_annealing_factor(min_anneal_factor,
                               anneal_update,
                               epoch, n_batch, batch_idx):
    n_updates = epoch * n_batch + batch_idx

    if anneal_update > 0 and n_updates < anneal_update:
        anneal_factor = min_anneal_factor + \
            (1.0 - min_anneal_factor) * (
                (n_updates / anneal_update)
            )
    else:
        anneal_factor = 1.0
    return anneal_factor


def nll_metric(output, target, mask):
    assert output.dim() == target.dim() == 3
    assert output.size() == target.size()
    assert mask.dim() == 2
    assert mask.size(1) == output.size(1)
    loss = nll_loss(output, target)  # (batch_size, time_step, input_dim)
    loss = mask * loss.sum(dim=-1)  # (batch_size, time_step)
    loss = loss.sum(dim=1, keepdim=True)  # (batch_size, 1)
    return loss


def kl_div_metric(output, target, mask):
    mu1, logvar1 = output
    mu2, logvar2 = target
    assert mu1.size() == mu2.size()
    assert logvar1.size() == logvar2.size()
    assert mu1.dim() == logvar1.dim() == 3
    assert mask.dim() == 2
    assert mask.size(1) == mu1.size(1)
    kl = kl_div(mu1, logvar1, mu2, logvar2)
    kl = mask * kl.sum(dim=-1)
    kl = kl.sum(dim=1, keepdim=True)
    return kl


def bound_eval(output, target, mask):
    x_recon, mu_q, logvar_q = output
    x, mu_p, logvar_p = target
    # batch_size = x.size(0)
    neg_elbo = nll_metric(x_recon, x, mask) + \
        kl_div_metric([mu_q, logvar_q], [mu_p, logvar_p], mask)
    # tsbn_bound_sum = elbo.div(mask.sum(dim=1, keepdim=True)).sum().div(batch_size)
    bound_sum = neg_elbo.sum().div(mask.sum())
    return bound_sum


def importance_sample(batch_idx, model, x, x_reversed, x_seq_lengths, mask, n_sample=500):
    sample_batch_size = 25
    n_batch = n_sample // sample_batch_size
    sample_left = n_sample % sample_batch_size
    if sample_left == 0:
        n_loop = n_batch
    else:
        n_loop = n_batch + 1

    ll_estimate = torch.zeros(n_loop).to(x.device)

    start_time = time.time()
    for i in range(n_loop):
        if i < n_batch:
            n_repeats = sample_batch_size
        else:
            n_repeats = sample_left

        x_tile = x.repeat_interleave(repeats=n_repeats, dim=0)
        x_reversed_tile = x_reversed.repeat_interleave(repeats=n_repeats, dim=0)
        x_seq_lengths_tile = x_seq_lengths.repeat_interleave(repeats=n_repeats, dim=0)
        mask_tile = mask.repeat_interleave(repeats=n_repeats, dim=0)

        x_recon, z_q_seq, z_p_seq, mu_q_seq, logvar_q_seq, mu_p_seq, logvar_p_seq = \
            model(x_tile, x_reversed_tile, x_seq_lengths_tile)

        q_dist = Normal(mu_q_seq, logvar_q_seq.exp().sqrt())
        p_dist = Normal(mu_p_seq, logvar_p_seq.exp().sqrt())
        log_qz = q_dist.log_prob(z_q_seq).sum(dim=-1) * mask_tile
        log_pz = p_dist.log_prob(z_q_seq).sum(dim=-1) * mask_tile
        log_px_z = -1 * nll_loss(x_recon, x_tile).sum(dim=-1) * mask_tile
        ll_estimate_ = log_px_z.sum(dim=1, keepdim=True) + \
            log_pz.sum(dim=1, keepdim=True) - \
            log_qz.sum(dim=1, keepdim=True)

        ll_estimate[i] = ll_estimate_.sum().div(mask.sum())

    ll_estimate = ll_estimate.sum().div(n_sample)
    print("%s-th batch, importance sampling took %.4f seconds." % (batch_idx, time.time() - start_time))

    return ll_estimate


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        self._data = self._data[self._data.counts != 0]
        return dict(self._data.average)

    def write_to_logger(self, key, value=None):
        assert self.writer is not None
        if value is None:
            self.writer.add_scalar(key, self._data.average[key])
        else:
            self.writer.add_scalar(key, value)


def neuralhmm_loss(x, x_hat, mu1, var1, mu2, var2, kl_annealing_factor=1, mask=None):
    kl_raw = kl_div(mu1, var1, mu2, var2)
    nll_raw = nll_loss(x_hat, x)
    # feature-dimension reduced
    kl_fr = kl_raw.mean(dim=-1)
    nll_fr = nll_raw.mean(dim=-1)
    # masking
    if mask is not None:
        mask = mask.gt(0).view(-1)
        kl_m = kl_fr.view(-1).masked_select(mask).mean()
        nll_m = nll_fr.view(-1).masked_select(mask).mean()
    else:
        kl_m = kl_fr.view(-1).mean()
        nll_m = nll_fr.view(-1).mean()

    loss = kl_m * kl_annealing_factor + nll_m

    return kl_raw, nll_raw, \
        kl_fr, nll_fr, \
        kl_m, nll_m, \
        loss


def gru_att(h_gru, t):
    att_score = torch.bmm(h_gru[:, 0:t, :], h_gru[:, t, :].unsqueeze(2)).squeeze(2) / math.sqrt(
        h_gru.shape[2])
    attention = Sparsemax(dim=1)(att_score)

    return attention


def print_metrics_binary(y_true, predictions, verbose=0):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        print("confusion matrix:")
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    f1_score=2*prec1*rec1/(prec1+rec1)
    if verbose:
        print("accuracy = {}".format(acc))
        print("precision class 0 = {}".format(prec0))
        print("precision class 1 = {}".format(prec1))
        print("recall class 0 = {}".format(rec0))
        print("recall class 1 = {}".format(rec1))
        print("AUC of ROC = {}".format(auroc))
        print("AUC of PRC = {}".format(auprc))
        print("min(+P, Se) = {}".format(minpse))
        print("f1_score = {}".format(f1_score))

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc,
            "minpse": minpse,
            "f1_score":f1_score}

def data_extract(epoch_data, best_epoch):
    if best_epoch == None:
        h_gru = epoch_data[0][0]
        mu_q = epoch_data[0][1]
        var_q = epoch_data[0][2]
        batch_x = epoch_data[0][3]
        batch_y = epoch_data[0][4]
        batch_name = epoch_data[0][5]
    else:
        h_gru = epoch_data[best_epoch][0]
        mu_q = epoch_data[best_epoch][1]
        var_q = epoch_data[best_epoch][2]
        batch_x = epoch_data[best_epoch][3]
        batch_y = epoch_data[best_epoch][4]
        batch_name = epoch_data[best_epoch][5]

    return h_gru, mu_q, var_q, batch_x, batch_y, batch_name
