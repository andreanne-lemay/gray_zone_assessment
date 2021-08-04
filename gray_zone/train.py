import os
import monai.transforms
import torch
import monai
from monai.transforms import Activations
from tqdm import tqdm
from monai.metrics import compute_roc_auc
from sklearn.metrics import cohen_kappa_score
import numpy as np
from sklearn.preprocessing import label_binarize

from gray_zone.loader import Dataset
from gray_zone.utils import get_label, get_validation_metric, modify_label_outputs_for_model_type
from gray_zone.models.coral import label_to_levels, proba_to_label


def train(model: [torch.Tensor],
          act: Activations,
          train_loader: Dataset,
          val_loader: Dataset,
          loss_function: any,
          optimizer: any,
          device: str,
          n_epochs: int,
          output_path: str,
          scheduler: any,
          n_class: int,
          model_type: str = 'classification',
          val_metric: str = None):
    """ Training loop. """
    best_metric = -np.inf
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    for epoch in range(n_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{n_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1

            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs, labels = modify_label_outputs_for_model_type(model_type, outputs, labels, act)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            val_loss = 0
            step = 0
            for val_data in tqdm(val_loader):
                val_images, val_labels = (
                    val_data[0].to(device),
                    val_data[1].to(device),
                )

                outputs = model(val_images)
                outputs, labels = modify_label_outputs_for_model_type(model_type, outputs, labels, act, val=True)

                val_loss += loss_function(outputs, val_labels)
                y_pred = torch.cat([y_pred, outputs], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                step += 1

            avg_val_loss = val_loss / step
            scheduler.step(val_loss / step)

            y_pred = y_pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            # Ordinal models require different data encoding
            if model_type == 'ordinal':
                y = get_label(y, arch_type=model_type, argmax=argmax)

            # Compute accuracy and validation metric
            y_pred_value = get_label(y_pred, arch_type=arch_type, argmax=argmax, n_class=n_class)
            acc_value = y_pred_value.flatten() == y.flatten()
            acc_metric = acc_value.sum() / len(acc_value)

            metric_value = get_validation_metric(val_metric, y_pred_value, y, y_pred, avg_val_loss, acc_metric)
            metric_values.append(metric_value)

            # If validation metric improves, save model
            if metric_value > best_metric:
                best_metric = metric_value
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    output_path, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current {val_metric}: {metric_value:.4f}"
                f" current accuracy: {acc_metric:.4f}"
                f" best {val_metric}: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
            # Save checkpoint
            torch.save(model.state_dict(), os.path.join(
                output_path, "checkpoints", f"checkpoint{epoch}.pth"))
