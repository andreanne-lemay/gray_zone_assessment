import os
import torchvision
import torch
import monai
from monai.transforms import Activations

from gray_zone.models import dropout_resnet, resnest, vit
from gray_zone.models.coral import CoralLayer


def get_model(architecture: str,
              model_type: str,
              dropout_rate: float,
              n_class: int,
              device: str,
              transfer_learning: str,
              img_dim: list or tuple):
    """ Init model """
    output_channels, act = get_model_type_params(model_type, n_class)
    if 'resnet' in architecture:
        resnet = getattr(dropout_resnet, architecture) if float(dropout_rate) > 0 else getattr(torchvision.models,
                                                                                               architecture)
        model = resnet(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, output_channels)
    elif 'resnest' in architecture:
        resnet = getattr(resnest, architecture)
        model = resnet(pretrained=True, final_drop=dropout_rate)
        model.fc = torch.nn.Linear(model.fc.in_features, output_channels)
    elif 'densenet' in architecture:
        densenet = getattr(monai.networks.nets, architecture)
        model = densenet(spatial_dims=2,
                         in_channels=3,
                         out_channels=output_channels,
                         dropout_prob=float(dropout_rate),
                         pretrained=True)
    elif 'vit' in architecture:
        model = vit.vit_b16(num_classes=output_channels, image_size=img_dim[1], dropout_rate=dropout_rate)
    else:
        raise ValueError("Only ResNet or Densenet models are available.")

    model = model.to(device)

    # Ordinal model requires a particular last layer to ensure coherent prediction (monotonic prediction)
    if model_type == 'ordinal':
        model = torch.nn.Sequential(
            model,
            CoralLayer(output_channels, n_class)
        )
        model = model.to(device)

    # Transfer weights if transfer learning
    if transfer_learning is not None:
        pretrained_dict = {k: v for k, v in torch.load(transfer_learning, map_location=device).items() if
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
