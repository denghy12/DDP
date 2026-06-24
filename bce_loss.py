import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(BCELoss, self).__init__()

        self.eps = eps
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets
        """

        # Calculating Probabilities
        x_softmax = self.softmax(x)
        xs_pos = x_softmax[:, 1, :]

        xs_neg = x_softmax[:, 0, :]
        y = y.reshape(-1)
        xs_pos = xs_pos.reshape(-1)
        xs_neg = xs_neg.reshape(-1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        return -loss.sum()