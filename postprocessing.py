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

def raw_sdf_func(segImg):
    Sdf = sitk.SignedMaurerDistanceMap(segImg, insideIsPositive=False, squaredDistance=False)
    seg = sitk.GetArrayFromImage(Sdf)

    return seg

def newsdf_post_processor(pred, main_region_th = 100000, sdf_th = 35, region_th = 2000):
    pred_test = pred.copy()
    mask_whole = np.zeros_like(pred_test)
    for anot in range(1, pred.max() + 1):
        print('i', anot)
        pred_single = np.zeros_like(pred_test)
        pred_single[pred_test == anot] = 1

        connected_label = label(pred_single, connectivity=pred_single.ndim)
        props = regionprops(connected_label)
        sorted_Props = sorted(props, key=lambda e: e.__getitem__('area'), reverse=True)

        mask_single = np.zeros_like(pred_test)
        index = None
        for idx_r, i in enumerate(range(len(sorted_Props))):
            print('ii', i)
            if sorted_Props[i]['area'] > main_region_th or idx_r == 0:
                print(sorted_Props[i]['area'])
                mask_single[connected_label == sorted_Props[i]['label']] = 1
            else:
                # only keep region bigger than main_region_th, here.
                index = i
                break
        if index == None:
            mask_whole[mask_single > 0] = 1
            continue

        ### second stage
        sdf_distance_mask_single = raw_sdf_func(sitk.GetImageFromArray(mask_single))
        sdf_mask_single = np.zeros_like(pred)
        sdf_mask_single[sdf_distance_mask_single < sdf_th] = 1
        sdf_mask_single = sdf_mask_single.astype('uint16')

        # if True:
        #     sdf_mask_single_2save = sdf_mask_single.copy() * 5
        #     sdf_mask_single_2save[pred > 0] = pred[pred > 0]
        #     path = '/home1/pbliu/all_data/nnUNet/IL_test_tmp/2017_05163189_SongSiBao_crop_mask_4label.nii.gz'
        #     _, _, meta = _sitk_Image_reader(path)
        #     print('name:', '_visual{}.nii.gz'.format(anot))
        #     _sitk_image_writer(sdf_mask_single_2save, meta, path.replace('.nii.gz', '_visual{}.nii.gz'.format(anot)))
        # else:
        #     pass

        for i in range(index, len(sorted_Props)):
            print('iii', i)
            if sorted_Props[i]['area'] < region_th:
                break
            else:
                part = np.zeros_like(pred_test)
                part[connected_label == sorted_Props[i]['label']] = 1
                if (part * sdf_mask_single).sum() > 0:
                    mask_single[connected_label == sorted_Props[i]['label']] = 1

        mask_whole[mask_single > 0] = 1
    result = mask_whole * pred
    return result


def maximum_connected_region_post_processor(pred, region_th = 100000):
    """
    pred: multi-label
    return: multi-label
    """
    pred_test = pred.copy()
    mask_whole = np.zeros_like(pred)

    for i in range(1, pred.max()+1):
        pred_single = np.zeros_like(pred_test)
        pred_single[pred_test==i] = 1

        connected_label = label(pred_single, connectivity=pred_single.ndim)
        props = regionprops(connected_label)
        sorted_Props = sorted(props, key=lambda e: e.__getitem__('area'), reverse=True)

        mask_single = np.zeros_like(pred)
        for i in range(len(sorted_Props)):
            # print(sorted_Props[i]['area'],sorted_Props[i]['label'])
            if sorted_Props[i]['area']>region_th or i==0: # i==0, make sure the biggest area can be keeped.
                mask_single[connected_label==sorted_Props[i]['label']] = 1
            else:
                pass

        mask_whole[mask_single>0]=1

    return mask_whole*pred

if __name__ == '__main__':
    """
    SDF post processor
    
    """
    def func(name, path, savepath, post='sdf'):
        _, image, meta = _sitk_Image_reader(path+'/'+name)
        print(name, image.shape)
        if post=='sdf':
            post_image = newsdf_post_processor(pred=image, region_th=2000, sdf_th=35)
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

