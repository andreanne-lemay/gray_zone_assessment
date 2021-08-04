import os
import torchvision
import torch
import monai
from monai.transforms import Activations

from gray_zone.models import dropout_resnet
from gray_zone.models.coral import CoralLayer


def get_model(architecture: str,
              model_type: str,
              dropout_rate: float,
              n_class: int,
              device: str,
              transfer_learning: bool,
              output_dir: str):
    """ Init model """
    output_channels, act = get_model_type_params(model_type, n_class)
    if 'resnet' in architecture:
        resnet = getattr(dropout_resnet, architecture) if float(dropout_rate) > 0 else getattr(torchvision.models,
                                                                                               architecture)
        model = resnet(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, out_channels)
        model = model.to(device)

    elif 'densenet' in architecture:
        densenet = getattr(monai.networks.nets, architecture)
        model = densenet(spatial_dims=2,
                         in_channels=3,
                         out_channels=out_channels,
                         dropout_prob=float(dropout_rate),
                         pretrained=True).to(device)
    else:
        raise ValueError("Only ResNet or Densenet models are available.")

    # Ordinal model requires a particular last layer to ensure coherent prediction (monotonic prediction)
    if model_type == 'ordinal':
        model = torch.nn.Sequential(
            model,
            CoralLayer(out_channels, n_class)
        )
        model = model.to(device)

    # Transfer weights if transfer learning
    if transfer_learning:
        best_model_path = os.path.join(output_dir, "best_metric_model.pth")
        pretrained_dict = {k: v for k, v in torch.load(best_model_path, map_location=device).items() if
                           k in model.state_dict()
                           and v.size() == model.state_dict()[k].size()}
        model.state_dict().update(pretrained_dict)
        model.load_state_dict(model.state_dict())

    return model, act


def get_model_type_params(model_type: str,
                          n_class: int):
    if model_type == 'ordinal':
        # Intermediate number of nodes
        out_channels = 10
        act = Activations(sigmoid=True)
    elif model_type == 'regression':
        out_channels = 1
        act = Activations(other=lambda x: x)
    elif model_type == 'classification':
        # Multiclass model
        if n_class > 2:
            act = Activations(softmax=True)
            out_channels = n_class
        # Binary model
        else:
            act = Activations(sigmoid=True)
            out_channels = 1
    else:
        raise ValueError("Model type needs to be 'ordinal', 'regression' or 'classification'.")

    return out_channels, act
