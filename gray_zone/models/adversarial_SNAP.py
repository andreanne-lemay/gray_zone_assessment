import numpy as np
import torch
from torch import nn


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
    and dividing by the dataset standard deviation.
    In order to certify radii in original coordinates rather than standardized coordinates, we
    add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
    layer of the classifier rather than as a part of preprocessing as is typical.
    """

    def __init__(self, means, sds):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', torch.tensor(means))
        self.register_buffer('sds', torch.tensor(sds))
        # self.means = torch.tensor(means) #.cuda()
        # self.sds = torch.tensor(sds)# .cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds


class NoiseLayer(nn.Module):
    """
    Main Class
    """

    def __init__(self, dimwise_noi_std_pt, unit_std_scale, m_dist):
        """
        Constructor
        """
        super().__init__()
        self.register_buffer('dimwise_noi_std_pt', dimwise_noi_std_pt)
        self.register_buffer('unit_std_scale', unit_std_scale)
        self.model_m_dist = m_dist

    def forward(self, x):
        unit_var_noise = torch.mul(self.unit_std_scale, torch.squeeze(self.model_m_dist.sample(x.data.size())))
        final_recon_x = x + torch.unsqueeze(self.dimwise_noi_std_pt, 0) * unit_var_noise

        return final_recon_x


def get_snap_net(model, device, img_size):

    # Use Laplacian noise since it showed the best results in the paper
    m_dist = torch.distributions.laplace.Laplace(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).to(device))
    unit_std_scale = torch.sqrt(torch.tensor([0.5]).to(device))

    # Choice from paper
    total_noi_pw = 4500
    img_size = np.array(img_size).transpose()
    dimwise_noi_var_all_np = (total_noi_pw / np.prod(img_size)) * np.ones(img_size.transpose())
    dimwise_noi_var_all = torch.from_numpy(dimwise_noi_var_all_np).float().to(device)
    dimwise_noi_std_pt = torch.sqrt(dimwise_noi_var_all)

    total_noi_pw_pt = torch.sum(dimwise_noi_var_all)

    normalize_layer = NormalizeLayer([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    InstNoiseLayer = NoiseLayer(dimwise_noi_std_pt, unit_std_scale, m_dist)

    return torch.nn.Sequential(InstNoiseLayer, normalize_layer, model), [total_noi_pw_pt, dimwise_noi_std_pt]


def _record_eta_batchwise(model, X, y, args):
    epsilon = args.epsilon_attack
    num_steps = args.num_steps_attack
    step_size = epsilon * 0.8

    X_pgd = torch.autograd.Variable(X.data, requires_grad=True)
    model.eval()
    if args.random_start:
        with torch.no_grad():
            random_noise = torch.FloatTensor(*X_pgd.shape).normal_(mean=0,
                                                                   std=2 * epsilon).detach().cuda()  # .uniform_(-epsilon, epsilon).to(device)
            random_noise_reshaped = random_noise.view(random_noise.size(0), -1)
            random_noise_reshaped_norm = torch.norm(random_noise_reshaped, p=2, dim=1, keepdim=True)
            all_epsilon_vec = (epsilon * torch.ones(
                [random_noise_reshaped_norm.size(0), random_noise_reshaped_norm.size(1)])).type_as(
                random_noise_reshaped_norm)
            random_noise_reshaped_normzed = epsilon * torch.div(random_noise_reshaped,
                                                                torch.max(random_noise_reshaped_norm,
                                                                          all_epsilon_vec).expand(-1,
                                                                                                  random_noise_reshaped.size(
                                                                                                      1)) + 1e-8)
            random_noise_final = random_noise_reshaped_normzed.view(X_pgd.size(0), X_pgd.size(1), X_pgd.size(2),
                                                                    X_pgd.size(3))

        X_pgd = Variable(X_pgd.data + random_noise_final.data, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            # loss = nn.CrossEntropyLoss()(model(X_pgd), y)
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
    return eta_final  # .cpu()


## this is your function for doing extra epoch to collect adv. perturbation statistics..
def eval_adv_train_whitebox(model, epoch_no, train_loader, args):
    model.eval()
    rob_err_train_tot = 0
    nat_err_train_tot = 0
    batch_count = 0

    for data, target in train_loader:
        batch_size = len(data)

        data, target = data.cuda(), target.cuda()
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        eta_final_batch = _record_eta_batchwise(model, X, y, args)
        print(batch_count)

        eta_mean_sq_proj_batch = torch.mean(torch.mul(eta_final_batch, eta_final_batch), dim=0)

        if 'stored_eta_mean_sq_proj_final' in locals():
            stored_eta_mean_sq_proj_final = stored_eta_mean_sq_proj_final + eta_mean_sq_proj_batch.data
        else:
            stored_eta_mean_sq_proj_final = eta_mean_sq_proj_batch.data

        assert not torch.isnan(eta_mean_sq_proj_batch).any()

        ### Here you can potentially break the loop...
        ## Run it only as a partial epoch..
        if batch_count == args.ssn_epoch_batches:
            break

        batch_count = batch_count + 1

    return stored_eta_mean_sq_proj_final
