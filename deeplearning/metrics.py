import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def get_acc(output, target):
    # takes in two tensors to compute accuracy
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    correct = pred.eq(target.data.view_as(pred)).cpu().sum()
    conf_mat = confusion_matrix(pred.cpu().numpy(), target.data.cpu().numpy(), labels=range(2))
    return np.squeeze(pred.cpu().numpy()), correct, target.size(0), conf_mat

def get_cohen_kappa_score(output, target):
    pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
    return(cohen_kappa_score(pred.cpu().numpy(), target.data.cpu().numpy()))

def get_cohen_kappa_score_from_confmat(confusion, weights=None):
    n_classes = confusion.shape[0]
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    if weights is None:
        w_mat = np.ones([n_classes, n_classes], dtype=np.int)
        w_mat.flat[:: n_classes + 1] = 0
    elif weights == "linear" or weights == "quadratic":
        w_mat = np.zeros([n_classes, n_classes], dtype=np.int)
        w_mat += np.arange(n_classes)
        if weights == "linear":
            w_mat = np.abs(w_mat - w_mat.T)
        else:
            w_mat = (w_mat - w_mat.T) ** 2
    else:
        raise ValueError("Unknown kappa weighting type.")

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    return(1 - k)