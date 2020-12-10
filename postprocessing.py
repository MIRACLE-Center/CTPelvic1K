from skimage.measure import label, regionprops
import numpy as np
import SimpleITK as sitk
from utils import _sitk_Image_reader, _sitk_image_writer
import os
from multiprocessing import Pool
from functools import partial

###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---
def gatherfiles(path, prefix=None, midfix=None, postfix=None, extname=True):
    files = os.listdir(path)
    if not prefix is None:
        files = [i for i in files if i.startswith(prefix)]
    if not midfix is None:
        files = [i for i in files if midfix in i]
    if not postfix is None:
        files = [i for i in files if i.endswith(postfix)]
    if extname:
        return files
    else:
        files = [os.path.splitext(i)[0] for i in files]
        return files

def sdf_func(segImg):
    """
    segImg is a sitk Image
    """
    Sdf = sitk.SignedMaurerDistanceMap(segImg, squaredDistance=False)
    # Sdf = sitk.Sigmoid(-Sdf, 50, 0, 1, 0)  # alpha, beta, max, min
    Sdf = sitk.Sigmoid(-Sdf, 10, 0, 1, 0)  # alpha, beta, max, min
    seg = sitk.GetArrayFromImage(Sdf)
    seg[seg > 0.4999] = 1  # convert sdf back to numpy array, and clip 0.5 above to 1 (inside)
    return seg

def oldsdf_post_processor(pred, main_region_th = 100000, sdf_th = 0.2, region_th = 2000):
    pred_single = pred.copy()
    pred_single[pred>1] = 1

    connected_label = label(pred_single, connectivity=pred_single.ndim)
    props = regionprops(connected_label)
    sorted_Props = sorted(props, key=lambda e: e.__getitem__('area'), reverse=True)
    mask = np.zeros_like(pred)

    index = None

    for i in range(len(sorted_Props)):
        if sorted_Props[i]['area'] > main_region_th or i==0:#zhu xin guer
        # if i==0:#zhu xin guer
            mask[connected_label==sorted_Props[i]['label']] = 1
        else:
            index = i
            break

    if index == None:
        return pred


    sdf_distance_mask = sdf_func(sitk.GetImageFromArray(mask))
    sdf_mask =sdf_distance_mask.copy()
    sdf_mask[sdf_mask > sdf_th] = 1
    sdf_mask = sdf_mask.astype('uint16')
    if False:
        visual = pred_single + sdf_mask*7
        _sitk_image_writer(visual, meta, path.replace('.nii.gz','_visual.nii.gz'))
    else:
        pass

    for i in range(index, len(sorted_Props)):
        if sorted_Props[i]['area'] < region_th:
            break
        else:
            part = np.zeros_like(pred)
            part[connected_label==sorted_Props[i]['label']]=1

            if (part*sdf_mask).sum() >0:
                mask[connected_label==sorted_Props[i]['label']] = 1
            else:
                pass

    result = mask*pred
    return result


def maximum_connected_region_post_processor(pred, region_th = 2000):
    """
    pred: multi-label
    return: multi-label
    """
    pred_single = pred.copy()
    pred_single[pred>1] = 1

    connected_label = label(pred_single, connectivity=pred_single.ndim)
    props = regionprops(connected_label)
    sorted_Props = sorted(props, key=lambda e: e.__getitem__('area'), reverse=True)
    mask = np.zeros_like(pred)
    for i in range(len(sorted_Props)):
        print(sorted_Props[i]['area'],sorted_Props[i]['label'])
        if sorted_Props[i]['area']>region_th or i==0: # i==0, make sure the biggest area can be keeped.
            mask[connected_label==sorted_Props[i]['label']] = 1
        else:
            pass

    return mask*pred

if __name__ == '__main__':
    """
    SDF post processor
    
    """
    def func(name, path, savepath, post='sdf'):
        _, image, meta = _sitk_Image_reader(path+'/'+name)
        print(name, image.shape)
        if post=='sdf':
            post_image = oldsdf_post_processor(pred=image, region_th=2000, sdf_th=0.25)
        elif post=='mcr':
            post_image = maximum_connected_region_post_processor(image, region_th=100000)
        else:
            raise NotImplementedError

        _sitk_image_writer(post_image, meta, savepath+'/'+name)

    base_dir = os.environ['HOME']
    pred_path = base_dir + "/all_data/nnUNet/rawdata/ipcai2021_ALL_Test/SDF_show" \
                "/Task22_ipcai2021_T__nnUNet_without_mirror_IPCAI2021_deeps_exclusion__nnUNet_without_mirror_IPCAI2021_deeps_exclusion__fold0_3dcascadefullres_pred"
    save_path = pred_path+'___newSDFpost'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print(pred_path)

    files = gatherfiles(pred_path, postfix='.nii.gz')

    print(len(files))
    files = list(set(files))
    print(len(files))
    files = sorted(files)

    pool = Pool(16)
    fu = partial(func, path=pred_path, savepath=save_path)
    _=pool.map(fu, files)
    pool.close()
    pool.join()

