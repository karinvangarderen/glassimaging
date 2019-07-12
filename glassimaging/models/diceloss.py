import torch.nn as nn
import torch

class DiceLoss(nn.Module):

    def __init__(self, n_classes=5, weights=(0.1,1,1,0,1)):
        super(DiceLoss, self).__init__()
        self.weights = torch.Tensor(list(weights))
        self.weights.requires_grad = False
        self.epsilon = torch.Tensor([0.00001])
        self.epsilon.requires_grad = False

    def forward(self, x, y):
        eps = self.epsilon.cuda(x.get_device()) if x.is_cuda else self.epsilon
        we = self.weights.cuda(x.get_device()) if x.is_cuda else self.weights
        loss = 0
        for i, w in enumerate(we):
            x_i = x[:, i]
            y_i = y[:, i]
            intersection = torch.sum(torch.mul(x_i, y_i))
            union = torch.add(torch.sum(x_i), torch.sum(y_i))
            union = torch.add(union, eps)

            loss = loss - torch.mul(torch.div(intersection, union), w)
        loss = torch.mul(loss, 2)
        loss = torch.div(loss, len(we))
        return loss

