import os
import monai.transforms
import torch
from torch.autograd import Variable
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
          val_metric: str = None,
          adv_noise: torch.Tensor = None,
          noise_params: list = None):
    """ Training loop. """
    best_metric = -np.inf
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    total_noi_pw_pt, dimwise_noi_std_pt = noise_params
    for epoch in range(n_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{n_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in tqdm(train_loader):
            step += 1

            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            # If adversarial training, apply noise to input
            if adv_noise is not None:
                # FreeTrain batch repeats
                n_repeat = 4
                for n in range(n_repeat):
                    noise_batch = Variable(adv_noise[0:inputs.size(0)], requires_grad=True).to(device)
                    in1 = inputs + noise_batch
                    in1.clamp_(0, 1.0)
                    optimizer.zero_grad()
                    outputs = model(in1)
                    outputs, labels = modify_label_outputs_for_model_type(model_type, outputs, labels, act, n_class)

                    loss = loss_function(outputs, labels)
                    loss.backward()
                    # Update perturbation
                    # fgsm: Fast Gradient Sign Method
                    # Values taken from https://github.com/adpatil2/SNAP
                    clip_eps = 0.0016
                    fgsm_step = 0.0016
                    pert = fgsm_step * torch.sign(noise_batch.grad)
                    adv_noise[0:inputs.size(0)] += pert.data
                    adv_noise.clamp_(-clip_eps, clip_eps)
                    optimizer.step()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs, labels = modify_label_outputs_for_model_type(model_type, outputs, labels, act, n_class)
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
                outputs, val_labels = modify_label_outputs_for_model_type(model_type, outputs, val_labels, act, n_class,
                                                                          val=True)

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
                y = get_label(y, model_type=model_type, n_class=n_class)

            # Compute accuracy and validation metric
            y_pred_value = get_label(y_pred, model_type=model_type, n_class=n_class)
            acc_value = y_pred_value.flatten() == y.flatten()
            acc_metric = acc_value.sum() / len(acc_value)

            metric_value = get_validation_metric(val_metric, y_pred_value, y, y_pred, avg_val_loss, acc_metric,
                                                 model_type, n_class)
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

        if (epoch % 5) == 0:
            stored_eta_mean_sq_proj = eval_adv_train_whitebox(model, epoch, train_loader)

            stored_eta_rt_mean_sq_proj = torch.sqrt(stored_eta_mean_sq_proj)
            normzed_eta_rt_mean_sq_proj = stored_eta_rt_mean_sq_proj / torch.sum(stored_eta_rt_mean_sq_proj)
            DimWise_noi_var_all = normzed_eta_rt_mean_sq_proj * total_noi_pw_pt
            dimwise_noi_std_pt = torch.sqrt(DimWise_noi_var_all)

            model.state_dict()['0.dimwise_noi_std_pt'].copy_(dimwise_noi_std_pt)

            print("updated noise std:")
            print(model.state_dict()['0.dimwise_noi_std_pt'])

            print("total pgd noise power (times no_of_batches)")
            print(torch.sum(stored_eta_mean_sq_proj))
            print("shape of dimwise_noi_std_pt is:")
            print(dimwise_noi_std_pt.size())


def eval_adv_train_whitebox(model, epoch_no, train_loader):
    model.eval()
    rob_err_train_tot = 0
    nat_err_train_tot = 0
    batch_count = 0

    for batch_data in train_loader:
        data, target = batch_data[0], batch_data[1]
        batch_size = len(data)

        data, target = data.cuda(), target.cuda()
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        eta_final_batch = _record_eta_batchwise(model, X, y)
        print(batch_count)

        eta_mean_sq_proj_batch = torch.mean(torch.mul(eta_final_batch, eta_final_batch), dim=0)

        if 'stored_eta_mean_sq_proj_final' in locals():
            stored_eta_mean_sq_proj_final = stored_eta_mean_sq_proj_final + eta_mean_sq_proj_batch.data
        else:
            stored_eta_mean_sq_proj_final = eta_mean_sq_proj_batch.data

        assert not torch.isnan(eta_mean_sq_proj_batch).any()

        ### Here you can potentially break the loop...
        ## Run it only as a partial epoch..
        if batch_count == 900:
            break

        batch_count = batch_count + 1

    return stored_eta_mean_sq_proj_final


def _record_eta_batchwise(model, X, y):
    # Default value
    epsilon = 4.0
    num_steps = 4
    step_size = epsilon * 0.8

    X_pgd = Variable(X.data, requires_grad=True)
    model.eval()

    with torch.no_grad():
        random_noise = torch.FloatTensor(*X_pgd.shape).normal_(mean=0,
                                                               std=2 * epsilon).detach().cuda()  # .uniform_(-epsilon, epsilon).to(device)
        random_noise_reshaped = random_noise.view(random_noise.size(0), -1)
        random_noise_reshaped_norm = torch.norm(random_noise_reshaped, p=2, dim=1, keepdim=True)
        all_epsilon_vec = (epsilon * torch.ones(
            [random_noise_reshaped_norm.size(0), random_noise_reshaped_norm.size(1)])).type_as(
            random_noise_reshaped_norm)
        random_noise_reshaped_normzed = epsilon * torch.div(random_noise_reshaped, torch.max(random_noise_reshaped_norm,
                                                                                             all_epsilon_vec).expand(-1,
                                                                                                                     random_noise_reshaped.size(
                                                                                                                         1)) + 1e-8)
        random_noise_final = random_noise_reshaped_normzed.view(X_pgd.size(0), X_pgd.size(1), X_pgd.size(2),
                                                                X_pgd.size(3))

    X_pgd = Variable(X_pgd.data + random_noise_final.data, requires_grad=True)

    for _ in range(num_steps):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = torch.nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        X_pgd_grad = X_pgd.grad.data
        with torch.no_grad():
            X_pgd_grad_reshaped = X_pgd_grad.view(X_pgd_grad.size(0), -1)
            X_pgd_grad_reshaped_norm = torch.norm(X_pgd_grad_reshaped, p=2, dim=1, keepdim=True)
            X_pgd_grad_reshaped_normzed = torch.div(X_pgd_grad_reshaped, X_pgd_grad_reshaped_norm.expand(-1,
                                                                                                         X_pgd_grad_reshaped.size(
                                                                                                             1)) + 1e-8)
            X_pgd_grad_normzed = X_pgd_grad_reshaped_normzed.view(X_pgd_grad.size(0), X_pgd_grad.size(1),
                                                                  X_pgd_grad.size(2), X_pgd_grad.size(3))
            eta = step_size * X_pgd_grad_normzed.data

            X_pgd = X_pgd.data + eta  # , requires_grad=True)  Variable(

            eta_tot = X_pgd.data - X.data

            eta_tot_reshaped = eta_tot.view(eta_tot.size(0), -1)
            eta_tot_reshaped_norm = torch.norm(eta_tot_reshaped, p=2, dim=1, keepdim=True)
            all_epsilon_vec = (
                        epsilon * torch.ones([eta_tot_reshaped_norm.size(0), eta_tot_reshaped_norm.size(1)])).type_as(
                eta_tot_reshaped_norm)
            eta_tot_reshaped_normzed = epsilon * torch.div(eta_tot_reshaped,
                                                           torch.max(eta_tot_reshaped_norm, all_epsilon_vec).expand(-1,
                                                                                                                    eta_tot_reshaped.size(
                                                                                                                        1)) + 1e-8)
            eta_tot_final = eta_tot_reshaped_normzed.view(X_pgd_grad.size(0), X_pgd_grad.size(1), X_pgd_grad.size(2),
                                                          X_pgd_grad.size(3))

        X_pgd = Variable(torch.clamp(X.data + eta_tot_final.data, 0, 1.0), requires_grad=True)
        # X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    with torch.no_grad():
        eta_final = X_pgd.data - X.data

    ## You need to remove this .cpu() on this line....
    return eta_final
