import torch
import torch.nn as nn

def Entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class Lim(nn.Module):
    def __init__(self, epsilon):
        super(Lim, self).__init__()
        self.epsilon = epsilon

    def forward(self, outs):
        softmax_out = nn.Softmax(dim=1)(outs)
        entropy_loss = torch.mean(Entropy(softmax_out))
        msoftmax = softmax_out.mean(dim=0)
        gentropy_loss = torch.sum(-msoftmax*torch.log(msoftmax+self.epsilon))
        entropy_loss -= gentropy_loss
        im_loss = entropy_loss * 1.0
        return im_loss
