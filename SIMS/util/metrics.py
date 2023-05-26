import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from pytorch_metric_learning.losses.ntxent_loss import NTXentLoss
from pytorch_metric_learning.utils import common_functions as c_f
import torch


def get_all_pairs_indices(labels, ref_labels=None):
    """
    Given a tensor of labels, this will return 4 tensors.
    The first 2 tensors are the indices which form all positive pairs
    The second 2 tensors are the indices which form all negative pairs
    """
    if ref_labels is None:
        ref_labels = labels
    labels1 = labels.unsqueeze(1)
    labels2 = ref_labels.unsqueeze(0)
    matches = (labels1 == labels2).byte()
    diffs = matches ^ 1
    if ref_labels is labels:
        matches.fill_diagonal_(0)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx


def convert_to_pairs(indices_tuple, labels):
    """
    This returns anchor-positive and anchor-negative indices,
    regardless of what the input indices_tuple is
    Args:
        indices_tuple: tuple of tensors. Each tensor is 1d and specifies indices
                        within a batch
        labels: a tensor which has the label for each element in a batch
    """
    if indices_tuple is None:
        return get_all_pairs_indices(labels)
    elif len(indices_tuple) == 4:
        return indices_tuple
    else:
        a, p, n = indices_tuple
        return a, p, a, n

class cont_NTXentLoss(NTXentLoss):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)
        self.label = None

    def update_label(self, label):
        self.label = label

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = convert_to_pairs(indices_tuple, labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings)
        return self.loss_method(mat, labels, indices_tuple)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, n = indices_tuple
        weight = 1
        if self.label is not None:
            weight = abs(self.label[a2] - self.label[n])

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            # print(torch.sum(neg_pairs))
            if self.label is not None:
                neg_pairs = neg_pairs * weight / 2
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)

            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()

class sds_NTXentLoss(NTXentLoss):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.add_to_recordable_attributes(list_of_names=["temperature"], is_stat=False)
        self.label = None

    def update_label(self, label):
        self.label = label

    def compute_loss(self, embeddings, labels, indices_tuple):
        indices_tuple = convert_to_pairs(indices_tuple, labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings)
        return self.loss_method(mat, labels, indices_tuple)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, n = indices_tuple
        weight = 1
        if self.label is not None:
            weight = abs(self.label[a2] - self.label[n])

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            # if dealing with actual distances, use negative distances
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            # print(torch.sum(neg_pairs))
            if self.label is not None:
                neg_pairs = neg_pairs * weight / 2
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = c_f.to_dtype(a2.unsqueeze(0) == a1.unsqueeze(1), dtype=dtype)
            neg_pairs = neg_pairs * n_per_p
            neg_pairs[n_per_p == 0] = c_f.neg_inf(dtype)

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)

            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()

def multiclass_acc(y_pred, y_true):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param y_pred: Float array representing the predictions, dimension (N,)
    :param y_true: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))


class Metrics:

    def eval_mosei_regression(self, y_true, y_pred, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = multiclass_acc(test_preds_a3, test_truth_a3)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_truth, non_zeros_binary_preds)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_truth, binary_preds)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        eval_results = {
            "Has0_acc_2": round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2": round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_3": round(mult_a3, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results

    def eval_sims_regression(self, y_true, y_pred):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        # non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        # non_zeros_binary_preds = (test_preds[non_zeros] > 0)
        #
        # non_zeros_acc2 = accuracy_score(non_zeros_binary_truth, non_zeros_binary_preds)
        # non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')


        # two classes{[-1.0, 0.0], (0.0, 1.0]}
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        # three classes{[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]}
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

        # five classes{[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]}
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a2 = np.sum(np.round(test_preds_a2) == np.round(test_truth_a2)) / float(len(test_truth_a2))
        mult_a3 = np.sum(np.round(test_preds_a3) == np.round(test_truth_a3)) / float(len(test_truth_a3))
        mult_a5 = np.sum(np.round(test_preds_a5) == np.round(test_truth_a5)) / float(len(test_truth_a5))

        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')

        eval_results = {
            "Mult_acc_2": round(mult_a2, 4),
            "Mult_acc_3": round(mult_a3, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "F1_score": round(f_score, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4),  # Correlation Coefficient
        }
        return eval_results
