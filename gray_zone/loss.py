import numpy as np
import torch
from gray_zone.models.coral import coral_loss
import numpy as np


class KappaLoss():
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, output, y):
        ohe_y = torch.zeros_like(output)
        batch_size = y.size(0)
        for i in range(batch_size):
            ohe_y[i, y[i].long()] = 1.

        output = torch.nn.Softmax(dim=1)(output)
        W = np.zeros((self.n_classes, self.n_classes))
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                W[i, j] = abs(i - j) ** 2

        W = torch.from_numpy(W.astype(np.float32)).to(y.device)

        O = torch.matmul(ohe_y.t(), output)
        E = torch.matmul(ohe_y.sum(dim=0).view(-1, 1), output.sum(dim=0).view(1, -1)) / O.sum()

        return (W * O).sum() / ((W * E).sum() + 1e-5)


def get_loss(loss_id: str,
             n_class: int):
    """ Get loss function from loss id. Choices between: 'ce', 'mse', 'l1', 'bce', 'mse', 'coral' """
    loss = None
    if loss_id == 'coral':
        loss = coral_loss
    elif loss_id == 'ce':
        loss = torch.nn.CrossEntropyLoss()
    elif loss_id == 'mse':
        loss = torch.nn.MSELoss()
    elif loss_id == 'qwk':
        loss = KappaLoss(n_classes=n_class)
    elif loss_id == 'l1':
        loss = torch.nn.L1Loss()
    elif loss_id == 'bce':
        loss = torch.nn.BCELoss()
    else:
        raise ValueError("Invalid loss function id. Choices: 'ce', 'mse', 'l1', 'bce', 'mse', 'coral'")

    return loss
