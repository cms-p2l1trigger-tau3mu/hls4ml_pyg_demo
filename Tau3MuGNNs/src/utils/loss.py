import torch
import torch.nn.functional as F


class Criterion(torch.nn.Module):

    def __init__(self, optimizer_config):
        super(Criterion, self).__init__()
        self.focal_loss = optimizer_config['focal_loss']
        self.alpha = optimizer_config['focal_alpha']
        self.gamma = optimizer_config['focal_gamma']

    def forward(self, inputs, targets):

        loss_dict = {}
        if self.focal_loss:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
            p_t = inputs * targets + (1 - inputs) * (1 - targets)
            loss = bce_loss * ((1 - p_t) ** self.gamma)

            if self.alpha >= 0:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                loss = alpha_t * loss
            loss = loss.mean()
            loss_dict['focal'] = loss.item()
        else:
            loss = F.binary_cross_entropy(inputs, targets)
            loss_dict['bce'] = loss.item()

        loss_dict['total'] = loss.item()
        return loss, loss_dict
