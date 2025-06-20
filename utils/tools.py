import os
import time
import random

import numpy as np

import shutil
from enum import Enum

import torch
import torchvision.transforms as transforms


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        

def load_model_weight(load_path, model, device, args):
    if os.path.isfile(load_path):
        print("=> loading checkpoint '{}'".format(load_path))
        checkpoint = torch.load(load_path, map_location=device)
        state_dict = checkpoint['state_dict']
        # Ignore fixed token vectors
        if "token_prefix" in state_dict:
            del state_dict["token_prefix"]

        if "token_suffix" in state_dict:
            del state_dict["token_suffix"]

        args.start_epoch = checkpoint['epoch']
        try:
            best_acc1 = checkpoint['best_acc1']
        except:
            best_acc1 = torch.tensor(0)
        if device is not 'cpu':
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(device)
        try:
            model.load_state_dict(state_dict)
        except:
            # TODO: implement this method for the generator class
            model.prompt_generator.load_state_dict(state_dict, strict=False)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(load_path, checkpoint['epoch']))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        print("=> no checkpoint found at '{}'".format(load_path))


def validate(val_loader, model, criterion, args, output_mask=None):
    # batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    # losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    # top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    # top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    # progress = ProgressMeter(
    #     len(val_loader),
    #     [batch_time, losses, top1, top5],
    #     prefix='Test: ')

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    top3 = AverageMeter('Acc@3', ':6.2f', Summary.AVERAGE)
    top10 = AverageMeter('Acc@10', ':6.2f', Summary.AVERAGE)
    top20 = AverageMeter('Acc@20', ':6.2f', Summary.AVERAGE)
    top30 = AverageMeter('Acc@30', ':6.2f', Summary.AVERAGE)
    top40 = AverageMeter('Acc@40', ':6.2f', Summary.AVERAGE)
    top50 = AverageMeter('Acc@50', ':6.2f', Summary.AVERAGE)
    top60 = AverageMeter('Acc@60', ':6.2f', Summary.AVERAGE)
    top65 = AverageMeter('Acc@65', ':6.2f', Summary.AVERAGE)
    top70 = AverageMeter('Acc@70', ':6.2f', Summary.AVERAGE)
    top75 = AverageMeter('Acc@75', ':6.2f', Summary.AVERAGE)
    top80 = AverageMeter('Acc@80', ':6.2f', Summary.AVERAGE)
    top85 = AverageMeter('Acc@85', ':6.2f', Summary.AVERAGE)
    top90 = AverageMeter('Acc@90', ':6.2f', Summary.AVERAGE)
    top95 = AverageMeter('Acc@95', ':6.2f', Summary.AVERAGE)
    top100 = AverageMeter('Acc@100', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5, top3, top10,top20,top30,top40,top50,top60,top65,top70,top75,top80,top85,top90,top95,top100],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                output = model(images)
                if output_mask:
                    output = output[:, output_mask]
                loss = criterion(output, target)

            # measure accuracy and record loss
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # losses.update(loss.item(), images.size(0))
            # top1.update(acc1[0], images.size(0))
            # top5.update(acc5[0], images.size(0))

            acc1, acc5, acc3, acc10, acc20, acc30, acc40, acc50, acc60, acc65, acc70, acc75, acc80, acc85, acc90, acc95, acc100 = accuracy(output, target, topk=(1, 5, 3, 10, 20,30,40,50,60,65,70,75,80,85,90,95,100))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            top3.update(acc3[0], images.size(0))
            top10.update(acc10[0], images.size(0))
            top20.update(acc20[0], images.size(0))
            top30.update(acc30[0], images.size(0))
            top40.update(acc40[0], images.size(0))
            top50.update(acc50[0], images.size(0))
            top60.update(acc60[0], images.size(0))
            top65.update(acc65[0], images.size(0))
            top70.update(acc70[0], images.size(0))
            top75.update(acc75[0], images.size(0))
            top80.update(acc80[0], images.size(0))
            top85.update(acc85[0], images.size(0))
            top90.update(acc90[0], images.size(0))
            top95.update(acc95[0], images.size(0))
            top100.update(acc100[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
        progress.display_summary()

    return top1.avg
