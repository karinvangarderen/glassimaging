import torch.nn as nn
import torch

class DiceLoss(nn.Module):

    def __init__(self, weights=(0.1,1,1,0,1)):
        super(DiceLoss, self).__init__()
        self.weights = torch.Tensor(list(weights))
        self.weights.requires_grad = False
        self.epsilon = torch.Tensor([1])
        self.epsilon.requires_grad = False

    def forward(self, x, y):
        eps = self.epsilon.cuda(x.get_device()) if x.is_cuda else self.epsilon
        we = self.weights.cuda(x.get_device()) if x.is_cuda else self.weights
        loss = 0
        y = y.long()
        y = y.unsqueeze(1)
        y_one_hot = torch.LongTensor(y.shape[0], len(we), y.shape[2], y.shape[3], y.shape[4]).zero_()
        y_one_hot = y_one_hot.cuda(x.get_device()) if x.is_cuda else y_one_hot
        y_one_hot = y_one_hot.scatter(1, y.data, 1)
        y_one_hot = y_one_hot.float()
        y_one_hot.requires_grad = False
        for i, w in enumerate(we):
            x_i = x[:, i]
            y_i = y_one_hot[:, i]
            intersection = torch.sum(torch.mul(x_i, y_i))
            intersection = torch.add(intersection, eps)
            union = torch.add(torch.sum(x_i), torch.sum(y_i))
            union = torch.add(union, eps)
            loss = loss - torch.mul(torch.div(intersection * 2, union), w)
        loss = torch.div(loss, torch.sum(we))
        return loss

