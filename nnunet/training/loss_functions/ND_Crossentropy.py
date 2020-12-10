import torch.nn
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
from .LovaszSoftmax import lovasz_softmax

class CrossentropyND(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def __init__(self):
        self.reduction = 'none'
        super(CrossentropyND, self).__init__(reduction=self.reduction)

    def forward(self, inp, target, heat_map=None):
        ### lovaszSoftmax loss ###
        # inp = F.softmax(inp, dim=1)
        # return lovasz_softmax(inp, target, ignore=255)

        ## raw implementation ###

        target = target.long()
        num_classes = inp.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)
        if heat_map is None:
            return  (super(CrossentropyND, self).forward(inp, target)).mean()
        else:
            heat_map = heat_map.view(-1,)

            return (super(CrossentropyND, self).forward(inp, target)*heat_map).mean()

        ###  sdf  heatmap ###
        loss_all = []
        for i in range(num_classes):
            target_index = torch.zeros_like(target).float()
            target_index[target==i] = 1
            inp_index = inp[:,i]
            loss_index = F.binary_cross_entropy_with_logits(inp_index, target_index, reduction=self.reduction).unsqueeze(1)
            loss_all.append(loss_index)

        loss_tmp = torch.cat(loss_all, 1)
        ### heatmap ###
        i0 = 1
        i1 = 2
        while i1 < len(heatmap.shape):  # this is ugly but torch only allows to transpose two axes at once
            heatmap = heatmap.transpose(i0, i1)
            i0 += 1
            i1 += 1

        heatmap = heatmap.contiguous()
        heatmap = heatmap.view(-1, num_classes)

        assert heatmap.shape == loss_tmp.shape, print(heatmap.shape, loss_tmp.shape)

        return (loss_tmp*heatmap.cuda()).mean()


class CrossentropyND_DeepS(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    """
    def __init__(self, weights):
        self.DSweights = weights
        self.reduction = 'none'
        super(CrossentropyND_DeepS, self).__init__(reduction=self.reduction)

    def forward(self, inps, targets, heat_map=None):
        inps = inps[:len(targets)] # resolution sorted from high to low
        # print('inps shape: ',[inp.shape for inp in inps])
        # print('tars shape: ',[tar.shape for tar in targets])
        targets = [target.long() for target in targets]
        num_classes = inps[0].size()[1]

        i0 = 1
        i1 = 2
        while i1 < len(inps[0].shape): # this is ugly but torch only allows to transpose two axes at once
            inps = [inp.transpose(i0, i1) for inp in inps]
            i0 += 1
            i1 += 1

        inps = [inp.contiguous() for inp in inps]
        inps = [inp.view(-1, num_classes) for inp in inps]

        targets = [target.view(-1,) for target in targets]

        losses = [super(CrossentropyND_DeepS, self).forward(inp, target) for inp, target in zip(inps, targets)]
        if not heat_map is None:
            heat_map = heat_map.view(-1,)
            losses[0] = losses[0]*heat_map

        losses = [loss.mean()*weight for loss, weight in zip(losses, self.DSweights)]
        return sum(losses)

# ------------------------------ sdf -----------------------------------------
def sdf_func(segImg):
    """
    segImg is a sitk Image
    """
    Sdf = sitk.SignedMaurerDistanceMap(segImg, squaredDistance=False)
    Sdf = sitk.Sigmoid(-Sdf, 50, 0, 1, 0)  # alpha, beta, max, min
    seg = sitk.GetArrayFromImage(Sdf)
    seg[seg > 0.4999] = 1  # convert sdf back to numpy array, and clip 0.5 above to 1 (inside)
    heat_map = seg + 0.5  # putGaussianMapF(mask, sigma=50.0)
    return heat_map

def sdf_func_convert(segImg):
    """
    segImg is a sitk Image
    """
    Sdf = sitk.SignedMaurerDistanceMap(segImg, squaredDistance=False)
    Sdf = sitk.Sigmoid(-Sdf, 50, 0, 1, 0)  # alpha, beta, max, min
    seg = sitk.GetArrayFromImage(Sdf)
    seg[seg > 0.4999] = 0.7  # convert sdf back to numpy array, and clip 0.5 above to 1 (inside)
    seg = 0.7-seg
    heat_map = seg + 1  # putGaussianMapF(mask, sigma=50.0)
    return heat_map

def get_converted_sdf_from_target(target):
    """

    :param target: (batch, c, h, w):(uint16)
    :return: (batch, c, h, w):(float32)
    """
    assert target.shape[1] == 1
    batch, _, h, w = target.shape
    heatmap = np.zeros((batch, 5, h, w), dtype='float32') ## float32 is important
    for i in range(5):
        sep_target = np.zeros_like(target) # b,c,h,w
        sep_target[target==i] = 1 # with one kind of target
        for slice in range(batch):
            slice_sep_target = sep_target[slice,:,:,:] # c,h,w
            Slice_Sep_Target = sitk.GetImageFromArray(slice_sep_target)
            heatmap[slice, i, :, :] = sdf_func_convert(Slice_Sep_Target)
    assert heatmap.max()>1.1

    return heatmap
def get_sdf_from_target(target):
    """

    :param target: (batch, c, h, w):(uint16)
    :return: (batch, c, h, w):(float32)
    """
    assert target.shape[1] == 1
    batch, _, h, w = target.shape
    heatmap = np.zeros((batch, 5, h, w), dtype='float32') ## float32 is important
    for i in range(5):
        sep_target = np.zeros_like(target) # b,c,h,w
        sep_target[target==i] = 1 # with one kind of target
        for slice in range(batch):
            slice_sep_target = sep_target[slice,:,:,:] # c,h,w
            Slice_Sep_Target = sitk.GetImageFromArray(slice_sep_target)
            heatmap[slice, i, :, :] = sdf_func(Slice_Sep_Target)
    assert heatmap.max()>1.1

    return heatmap

if __name__ == '__main__':
    path = '/Volumes/Pengbo_Kani/ICTDATA/kits19/nnunet_results/Spa_2_OK/new_spc_2/get_rid_of_pixel/new_case_00202.nii.gz'
    im = sitk.GetArrayFromImage(sitk.ReadImage(path))
    im = im[:,None]
    heat_map = get_converted_sdf_from_target(im)
    Im = sitk.GetImageFromArray(heat_map)
    sitk.WriteImage(Im, '/Users/pengbo/Desktop/aa.nii.gz')