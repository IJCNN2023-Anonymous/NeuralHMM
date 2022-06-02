import torch
import torch.nn as nn
from sparsemax import Sparsemax
from sklearn import metrics


def reparameterization(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_div(mu1, logvar1, mu2=None, logvar2=None):
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
    if logvar2 is None:
        logvar2 = torch.zeros_like(mu1)

    return 0.5 * (
        logvar2 - logvar1 + (
            torch.exp(logvar1) + (mu1 - mu2).pow(2)
        ) / torch.exp(logvar2) - 1)


def nll_loss(x_hat, x):
    assert x_hat.dim() == x.dim() == 3
    assert x.size() == x_hat.size()
    return nn.BCEWithLogitsLoss(reduction='none')(x_hat, x)


def hmmnn_loss(x, x_hat, mu1, logvar1, mu2, logvar2, kl_annealing_factor=1, mask=None):
    kl_raw = kl_div(mu1, logvar1, mu2, logvar2)
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

