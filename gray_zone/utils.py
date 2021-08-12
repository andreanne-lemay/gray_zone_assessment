from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    Transpose,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    NormalizeIntensity,
    Resize,
    ToTensor,
    AddChannel,
    CenterSpatialCrop,
    RandShiftIntensity,
    RandStdShiftIntensity,
    RandScaleIntensity,
)
from torchvision.transforms import Normalize, Lambda
import torch
import numpy as np
from monai.metrics import compute_roc_auc
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import label_binarize

from gray_zone.models.coral import proba_to_label, label_to_levels


def load_transforms(transforms_dict: dict):
    """Converts dictionary into python transforms"""
    transform_lst = []
    for tr in transforms_dict:
        transform_lst.append(globals()[tr](**transforms_dict[tr]))

    return Compose(transform_lst)


def get_label(pred: np.ndarray,
              model_type: str,
              n_class: int) -> np.ndarray:
    """ From model predictions get discrete label. """
    if model_type == 'ordinal':
        return proba_to_label(pred)

    elif model_type == 'regression':
        # Get predicted label by equally splitting the range of prediction
        # E.g. min pred = 0, max pred = 2. 0-0.66: class 0, 0.67-1.33: class 1, 1.33-2: class 2
        gt_max = max(pred.max(), n_class - 1)
        gt_min = min(pred.min(), 0)
        c_pred = []
        thr = [gt_min]
        for c in range(n_class - 1):
            new_thr = thr[c] + (gt_max-gt_min) / n_class
            thr.append(new_thr)
        thr.append(gt_max)
        for s in pred:
            for t in range(n_class):
                if (s[0] <= thr[t + 1]) and (s[0] >= thr[t]):
                    c_pred.append(t)
                    break
        return np.array(c_pred)

    else:
        return pred.argmax(axis=1)


def get_validation_metric(val_metric: str,
                          y_pred_label: np.ndarray,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          val_loss: any,
                          acc_metric: float,
                          model_type: str,
                          n_class: int = None):
    """ From validation metric id, return validation metric value. """
    if val_metric == 'kappa':
        metric_value = cohen_kappa_score(y_pred_label, y_true, weights='linear')
    elif val_metric == 'auc':
        metric_value = 0
        n_class = y_pred.shape[-1]
        for i in range(n_class):
            metric_value += compute_roc_auc(torch.tensor(y_pred[:, i]),
                                            torch.tensor(label_binarize(y_true, classes=np.arange(n_class))[:, i]))
        metric_value /= n_class
    elif val_metric == 'val_loss':
        metric_value = -val_loss.item()
    # Default validation metric accuracy
    elif val_metric == 'custom_cervix':
        if model_type == 'regression':
            if n_class == 2:
                metric_value = compute_roc_auc(torch.tensor(y_pred), torch.tensor(y_true))
            else:
                norm_score = y_pred
                multi_score = np.zeros((len(norm_score), 3))
                multi_score[:, 0] = 1 - norm_score.flatten()
                multi_score[:, 2] = norm_score.flatten()
                metric_value = (compute_roc_auc(torch.tensor(multi_score[:, 0]),
                                                torch.tensor(label_binarize(y_true, classes=list(range(n_class)))[:, 0])) +
                                compute_roc_auc(torch.tensor(multi_score[:, n_class - 1]),
                                                torch.tensor(label_binarize(y_true, classes=list(range(n_class)))[:,
                                                             n_class - 1]))) / 2
        elif model_type == 'ordinal':

            score = np.zeros((y_pred.shape[0], 3))
            for p_idx, p in enumerate(y_pred):
                score[p_idx, 0] = 1 - p[0]
                if p[1] <= 0.5:
                    score[p_idx, 1] = p[0]
                else:
                    score[p_idx, 1] = 1 - p[0]

                score[p_idx, 2] = p[1]

            pred = score / score.sum(1)[..., None]

            metric_value = (compute_roc_auc(torch.tensor(pred[:, 0]),
                                            torch.tensor(label_binarize(y_true, classes=list(range(n_class)))[:, 0])) +
                            compute_roc_auc(torch.tensor(pred[:, n_class - 1]),
                                            torch.tensor(
                                                label_binarize(y_true, classes=list(range(n_class)))[:, n_class - 1]))) / 2
        else:
            if y_pred.shape[1] > 1:
                metric_value = (compute_roc_auc(torch.tensor(y_pred[:, 0]),
                                                torch.tensor(label_binarize(y_true, classes=list(range(n_class)))[:, 0])) +
                                compute_roc_auc(torch.tensor(y_pred[:, n_class - 1]),
                                                torch.tensor(label_binarize(y_true, classes=list(range(n_class)))[:,
                                                             n_class - 1]))) / 2
            else:
                metric_value = compute_roc_auc(torch.tensor(y_pred), torch.tensor(y_true))

    else:
        metric_value = acc_metric

    return metric_value


def modify_label_outputs_for_model_type(model_type: str,
                                        outputs: any,
                                        labels: torch.Tensor,
                                        act: Activations,
                                        n_class: int,
                                        val: bool = False):
    """ Some model type requires specific data format. This functions modifies the label and model output to accomodate
    these cases. """
    # Convert GT into ordinal format
    if model_type == 'ordinal':
        labels = label_to_levels(labels, num_classes=n_class)
    if model_type == 'regression' or outputs.shape[-1] == 1:
        labels = labels.type(torch.float32)[..., None]

    # Activate output (softmax, sigmoid, etc.) during validation or for binary models
    if val or outputs.shape[-1] == 1:
        outputs = act(outputs)
    return outputs, labels
