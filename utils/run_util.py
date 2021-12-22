# from sklearn.metrics import confusion_matrix
import torch
from torch import optim

def get_optim_and_scheduler(params, epochs, lr, nesterov=False):
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    return optimizer, scheduler

def to_cuda(*parms):
    if not torch.cuda.is_available():
        return parms
    results = []
    for item in parms:
        results.append(item.cuda())
    return results

def shape_data(*datas):
    result = []
    for data in datas:
        size = [-1]
        size.extend(data.size()[2:])
        result.append(data.reshape(size))
    return result

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    now_lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        now_lr = param_group["lr"]
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer,now_lr
