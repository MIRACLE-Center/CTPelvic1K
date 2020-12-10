import torch
import numpy as np
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.training.loss_functions.ND_Crossentropy import CrossentropyND, CrossentropyND_DeepS
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., square=False):
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DC_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target, sdf_heatmap):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)
        return result


def reverse_gt(target_onehot):
    reversed_gt_onehot = torch.zeros_like(target_onehot)
    for i in range(target_onehot.shape[1]):
        reversed_gt_onehot[:, i] = 1 - target_onehot[:, i]
    return reversed_gt_onehot

def _convert_target2onehot(input, target):
    """
    :param input: pred
    :param target: long dtype
    :return:
    """
    target_onehot = torch.zeros_like(input)
    target_onehot.scatter_(1, target, 1)
    return target_onehot

class Exclusion_loss(nn.Module):
    def __init__(self, union_func):
        super(Exclusion_loss, self).__init__()
        self.union = union_func

    def forward(self, network_output, target):
        return -self.union(network_output, target)

class DC_and_CE_Exclusion_loss(DC_and_CE_loss):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", ex=True, rate=2):
        super(DC_and_CE_Exclusion_loss, self).__init__(soft_dice_kwargs, ce_kwargs, aggregate)
        self.ex = Exclusion_loss(self.dc)
        self.ex_choice = ex
        self.rate = rate
        assert self.rate>=0

    def forward(self, net_output, target, sdf_heatmap):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target, sdf_heatmap)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son")

        target_onehot = _convert_target2onehot(net_output, target.long())
        not_gt = reverse_gt(target_onehot)
        ex_loss = self.ex(net_output, not_gt)

        if self.ex_choice:
            result = result + self.rate * ex_loss
        return result

# --------------------------- deep s ----------------------------------
class SoftDiceLoss_DeepS(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., square=False, weights=None):
        super(SoftDiceLoss_DeepS, self).__init__()
        self.DSweights = weights
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, xs, ys, loss_mask=None):
        xs = xs[:len(ys)]
        shp_x = xs[0].shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            xs = [self.apply_nonlin(x) for x in xs]

        dcs = []
        for x, y in zip(xs, ys):
            tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

            dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

            if not self.do_bg:
                if self.batch_dice:
                    dc = dc[1:]
                else:
                    dc = dc[:, 1:]
            dc = dc.mean()
            dcs.append(dc)
        dcs = [dc * weight for dc, weight in zip(dcs, self.DSweights)]
        # print("dice loss device: ",sum(dcs).device)
        return -sum(dcs)/len(dcs)

class DownPooling(nn.Module):
    def __init__(self, deepspool):
        super(DownPooling, self).__init__()
        if len(deepspool[0]) == 3:
            self.pool0 = nn.MaxPool3d(kernel_size=deepspool[0], stride=deepspool[0])
            self.pool1 = nn.MaxPool3d(kernel_size=deepspool[1], stride=deepspool[1])
            self.pool2 = nn.MaxPool3d(kernel_size=deepspool[2], stride=deepspool[2])
        elif len(deepspool[0]) == 2:
            self.pool0 = nn.MaxPool2d(kernel_size=deepspool[0], stride=deepspool[0])
            self.pool1 = nn.MaxPool2d(kernel_size=deepspool[0], stride=deepspool[0])
            self.pool2 = nn.MaxPool2d(kernel_size=deepspool[0], stride=deepspool[0])

    def forward(self, target):
        target0 = self.pool0(target)
        target1 = self.pool1(target0)
        target2 = self.pool2(target1)

        return [target, target0,target1,target2]

class DC_and_CE_Exclusion_loss_DeepS(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", ex=True, rate=2, deepspool=None):
        super(DC_and_CE_Exclusion_loss_DeepS, self).__init__()
        self.down_gt = DownPooling(deepspool) # not a good implementation of this calculation. Upsample pred maybe a good attempt.

        self.aggregate = aggregate
        self.ce = CrossentropyND_DeepS(**ce_kwargs)
        self.dc = SoftDiceLoss_DeepS(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.ex = Exclusion_loss(self.dc)
        self.ex_choice = ex
        self.rate = rate
        assert self.rate >= 0
    def forward(self, net_outputs, target, sdf_heatmap):
        targets = self.down_gt(target)

        ce_loss = self.ce(net_outputs, targets, sdf_heatmap)
        dc_loss = self.dc(net_outputs, targets)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son")


        if self.ex_choice:
            target_onehots = [_convert_target2onehot(net_output, target.long()) for net_output, target in zip(net_outputs, targets)]
            not_gts = [reverse_gt(target_onehot) for target_onehot in target_onehots]
            ex_loss = self.ex(net_outputs, not_gts)

            result = result + self.rate * ex_loss
        return result
if __name__ == '__main__':
    dow = DownPooling([[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]])

    ipt = torch.zeros(2,5,64,64,64)
    print(ipt.shape)
    resuts = dow(ipt)

