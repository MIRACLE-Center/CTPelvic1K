import numpy as np
import SimpleITK as sitk
import time
import os
from utils import _sitk_Image_reader, _sitk_image_writer, save_pkl
from functools import partial
from multiprocessing import Pool
from collections import OrderedDict
from postprocessing import oldsdf_post_processor, maximum_connected_region_post_processor

def write2singlefile(content, savepath):
    with open(savepath,'a+') as f:
        f.write(content)

def computeQualityMeasures(lP, lT):
    """
    Binary [0,1]
    """
    quality = dict()
    try:
        labelPred = sitk.GetImageFromArray(lP, isVector=False)
        labelTrue = sitk.GetImageFromArray(lT, isVector=False)
        hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
        hausdorffcomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
        quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
        quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

        dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
        dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
        quality["dice"] = dicecomputer.GetDiceCoefficient()

        quality["pred_pixel_num"] = len(lP[lP==1])
        quality["target_pixel_num"] = len(lT[lT==1])
        return quality
    except Exception as e:
        quality = dict()
        print('Exception: ',e)
        quality["avgHausdorff"]=0
        quality["Hausdorff"]=0
        quality["dice"] = 1
        quality["pred_pixel_num"] = 0
        quality["target_pixel_num"] = 0
        return quality

def computeQualityMeasures_oneCases(name, pred_path, target_path_file, postprocessor, region_th=2000, sdf_th = 0.4):
    """
    need modified to suited format
    """
    _, pred, _ = _sitk_Image_reader(os.path.join(pred_path, name+'.nii.gz'))
    _, target, _ = _sitk_Image_reader(os.path.join(target_path_file, name+'_mask_4label.nii.gz'))
    print("computing {} ...".format(name), np.unique(target))
    # write2singlefile("computing {} ...".format(name)+ str(np.unique(target))+'\n', LOG_save_path)

    """
    Come on!!!!!!
    
    """
    if postprocessor == 'sdf':
        pred = oldsdf_post_processor(pred, sdf_th=sdf_th, region_th = region_th)
    elif postprocessor == 'mcr':
        pred = maximum_connected_region_post_processor(pred, region_th=region_th)
    elif postprocessor is None:
        pass
    else:
        raise NotImplementedError

    one_case_qualities = OrderedDict()

    range_right = target.max()+1
    for i in range(1, range_right):
        class_pred = np.zeros_like(pred)
        class_target = np.zeros_like(target)
        class_pred[pred == i] = 1
        class_target[target == i] = 1

        class_quality = computeQualityMeasures(class_pred, class_target)
        one_case_qualities[i] = class_quality


    pred[pred>1] = 1
    target[target>1 ]= 1
    assert len(np.unique(pred)) == 2
    assert len(np.unique(target)) == 2
    one_case_qualities['whole'] = computeQualityMeasures(pred, target)
    del pred, target


    hausdorffs = [one_case_qualities[i]["Hausdorff"] for i in range(1,range_right)]
    dices      = [one_case_qualities[i]["dice"] for  i in range(1,range_right)]
    pixel_nums = [one_case_qualities[i]["target_pixel_num"] for i in range(1,range_right)]
    hausdorffs = np.array(hausdorffs)
    dices      = np.array(dices)
    pixel_nums = np.array(pixel_nums)

    mean_hausdorff          = hausdorffs.mean()
    weighted_mean_hausdorff = (pixel_nums*hausdorffs).sum()/pixel_nums.sum()
    mean_dice               = dices.mean()
    weighted_mean_dice      = (pixel_nums*dices).sum()/pixel_nums.sum()

    one_case_qualities["mean_hausdorff"]          = mean_hausdorff
    one_case_qualities["mean_dice"]               = mean_dice
    one_case_qualities["weighted_mean_hausdorff"] = weighted_mean_hausdorff
    one_case_qualities["weighted_mean_dice"]      = weighted_mean_dice

    for i in range(range_right, 5):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {}: {} tianjia !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(name, i))
        one_case_qualities[i]=dict()
        one_case_qualities[i]["Hausdorff"]=0
        one_case_qualities[i]["dice"] = 1
    print(name,'\n',
          '1: ',one_case_qualities[1],'\n',
          '2: ',one_case_qualities[2],'\n',
          '3: ',one_case_qualities[3],'\n',
          '4: ',one_case_qualities[4],'\n',
          'mean_hausdorff: ',one_case_qualities['mean_hausdorff'],'\n',
          'mean_dice: ',one_case_qualities['mean_dice'],'\n',
          '-'*33,'\n')

    return (name, one_case_qualities)


def computeQualityMeasures_oneModel(pred_path, target_path_file, subdataset, postprocessor, thread, region_th,sdf_th):
    """
    pred_path:         prediction path
    target_path_file:  ground truth
    subdataset:        dataset1, dataset2...
    postprocessor:     sdf, mcr

    results structure:
    dict{patient:{1:quality,
                  2:quality,
                  ...}}
    """
    print('pred_path: ', pred_path)
    files = os.listdir(pred_path)

    '''
    need modify to suited format
    '''
    names = [i.replace('.nii.gz','') for i in files if i.endswith('.nii.gz') and not i.endswith('_metal.nii.gz')]

    if subdataset=='all':
        pass
    elif subdataset.startswith('dataset'):
        names = [i for i in names if i.startswith(subdataset)]
    else:
        raise EOFError

    print(names)
    print(pred_path)
    print('names: ',len(names))
    print('post: ',postprocessor)
    print('thread: ', thread)
    print('thresh:',region_th,sdf_th)

    """
    Come on!!!!!!
    """
    pool = Pool(thread)
    func = partial(computeQualityMeasures_oneCases, pred_path = pred_path, target_path_file = target_path_file, postprocessor = postprocessor,region_th=region_th,sdf_th=sdf_th)
    results = pool.map(func, names)
    pool.close()
    pool.join()

    di = OrderedDict()
    idx = 0
    mean_1_dice = 0
    mean_1_haud = 0
    mean_2_dice = 0
    mean_2_haud = 0
    mean_3_dice = 0
    mean_3_haud = 0
    mean_4_dice = 0
    mean_4_haud = 0
    mwhole_dice = 0
    mwhole_haud = 0
    mean_dice   = 0
    mean_haud   = 0
    w_mean_dice = 0
    w_mean_haud = 0
    for name, quailty in results:
        di[name] = quailty
        idx += 1
        mean_1_dice += quailty[1]["dice"]
        mean_2_dice += quailty[2]["dice"]
        mean_3_dice += quailty[3]["dice"]
        mean_4_dice += quailty[4]["dice"]
        mwhole_dice += quailty["whole"]["dice"]
        mean_1_haud += quailty[1]["Hausdorff"]
        mean_2_haud += quailty[2]["Hausdorff"]
        mean_3_haud += quailty[3]["Hausdorff"]
        mean_4_haud += quailty[4]["Hausdorff"]
        mwhole_haud += quailty["whole"]["Hausdorff"]

        mean_dice   += quailty["mean_dice"]
        mean_haud   += quailty["mean_hausdorff"]
        w_mean_dice += quailty["weighted_mean_dice"]
        w_mean_haud += quailty["weighted_mean_hausdorff"]

    print("mean_1_dice: ", mean_1_dice / idx)
    print("mean_1_huad: ", mean_1_haud / idx)
    print("mean_2_dice: ", mean_2_dice / idx)
    print("mean_2_huad: ", mean_2_haud / idx)
    print("mean_3_dice: ", mean_3_dice / idx)
    print("mean_3_huad: ", mean_3_haud / idx)
    print("mean_4_dice: ", mean_4_dice / idx)
    print("mean_4_huad: ", mean_4_haud / idx)
    print("mwhole_dice: ", mwhole_dice / idx)
    print("mwhole_haud: ", mwhole_haud / idx)
    print("mean_dice:   ", mean_dice / idx)
    print("mean_haud:   ", mean_haud / idx)
    print("w_mean_dice: ", w_mean_dice / idx)
    print("w_mean_haud: ", w_mean_haud / idx)
    print(pred_path)
    print('post: ',postprocessor)

    if postprocessor.endswith('sdf'):
        pklsave = os.path.join(pred_path, "evaluation_{}_{}__{}.pkl".format(postprocessor,sdf_th,region_th))
    else:
        pklsave = os.path.join(pred_path, "evaluation_{}__{}.pkl".format(postprocessor, region_th))
    save_pkl(di, pklsave)

    print(pklsave,'saved...')


if __name__ == '__main__':
    t_begin = time.time()
    predbasePath = os.path.join(os.environ['HOME'],'all_data/nnUNet/rawdata/ipcai2021_ALL_Test/')
    tarPath      = os.path.join(os.environ['HOME'],'all_data/nnUNet/rawdata/ipcai2021/')

    print(predbasePath)
    for fo in [0]:
        computeQualityMeasures_oneModel(
            pred_path=predbasePath+
                      f'Task22_ipcai2021_T__nnUNet_without_mirror_IPCAI2021_deeps_exclusion__nnUNet_without_mirror_IPCAI2021_deeps_exclusion__fold{fo}_3dcascadefullres_pred',
            target_path_file=tarPath,
            subdataset='all',
            postprocessor='mcr',
            thread=64,
            region_th=2000,
            sdf_th=0.25)

        t_end = time.time()
        print(f'time consuming {t_end-t_begin} s ...')
