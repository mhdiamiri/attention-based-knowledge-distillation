import torch
import numpy as np
import torch.nn as nn

LAYER = {
         'wrn40x2': np.arange(1, (40 - 4) // 2 + 1),  # 18
         'wrn16x2': np.arange(1, (16 - 4) // 2 + 1),  # 6
     }


def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes


def train_kl(module_list, optimizer, criterion, train_loader, device, args):
    for module in module_list:
        module.train() # set to train mode
    module_list[-1].eval() # set teacher to eval mode

    model_s = module_list[0]
    model_t = module_list[-1]

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion_ce, criterion_kl, criterion_kd = criterion
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device) # move input and target to device
        with torch.no_grad():
            feat_t, output_t = model_t(inputs, is_feat=True)
            feat_t = [f.detach() for f in feat_t] # only tensors
        feat_s, output_s = model_s(inputs, is_feat=True)

        loss_ce = criterion_ce(output_s, targets) # crossentropy
        #loss_kl = criterion_kl(output_s, output_t)
        loss_kd = criterion_kd(feat_s, feat_t) #L afd

        loss = loss_ce  + args.beta * loss_kd #args.alpha * loss_kl

        optimizer.zero_grad() # reset to zero
        loss.backward()
        # update
        optimizer.step()

        acc1, acc5 = accuracy(output_s, targets, topk=(1, 5))

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1, batch_size)
        top5.update(acc5, batch_size)
        # Top-5 means correct class is among top 5 predicted classes
        
    return losses.avg, top1.avg, top5.avg


def test(model, test_loader, device):
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

            batch_size = targets.size(0)
            top1.update(acc1, batch_size)
            top5.update(acc5, batch_size)

    return top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
