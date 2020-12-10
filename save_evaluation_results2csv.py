import pandas as pd
import pickle as pkl
import numpy as np
import os


if __name__ == '__main__':
    base_dir = os.environ['HOME']
    eval_reslult_pkl_path  = base_dir + '/all_data/nnUNet/rawdata/ipcai2021_ALL_Test/' \
                             'Task22_ipcai2021_T__nnUNet_without_mirror_IPCAI2021_deeps_exclusion__nnUNet_without_mirror_IPCAI2021_deeps_exclusion__fold0_3dcascadefullres_pred/' \
                             'evaluation_mcr__2000_False.pkl'

    print(eval_reslult_pkl_path)
    with open(eval_reslult_pkl_path,'rb') as f:
        eval_reslult = pkl.load(f) # dict of names and quality sub-dict

    datasets = ['dataset1','dataset2','dataset3','dataset4','dataset5','dataset6']
    for dataset in datasets:
        names = []
        bone1_Hausdorff = []
        bone1_Dice = []
        bone2_Hausdorff = []
        bone2_Dice = []
        bone3_Hausdorff = []
        bone3_Dice = []
        bone4_Hausdorff = []
        bone4_Dice = []
        whole_Hausdorff = []
        whole_Dice = []
        mean_Hausdorff = []
        mean_Dice = []
        weight_Hausdorff = []
        weight_Dice = []
        for na, quality in eval_reslult.items():
            if not dataset in na:
                continue
            names.append(na)
            bone1_Hausdorff.append(quality[1]['Hausdorff'])
            bone1_Dice.append(quality[1]['dice'])
            bone2_Hausdorff.append(quality[2]['Hausdorff'])
            bone2_Dice.append(quality[2]['dice'])
            bone3_Hausdorff.append(quality[3]['Hausdorff'])
            bone3_Dice.append(quality[3]['dice'])
            bone4_Hausdorff.append(quality[4]['Hausdorff'])
            bone4_Dice.append(quality[4]['dice'])
            whole_Hausdorff.append(quality['whole']['Hausdorff'])
            whole_Dice.append(quality['whole']['dice'])
            mean_Hausdorff.append(quality['mean_hausdorff'])
            mean_Dice.append(quality['mean_dice'])
            weight_Hausdorff.append(quality['weighted_mean_hausdorff'])
            weight_Dice.append(quality['weighted_mean_dice'])
        print(dataset, len(names))
        print(
            'bone1_Dice:', np.array(bone1_Dice).mean(),'\n',
         'bone1_Hausdorff:', np.array(bone1_Hausdorff).mean(),'\n',
         'bone2_Dice:', np.array(bone2_Dice).mean(),'\n',
         'bone2_Hausdorff:', np.array(bone2_Hausdorff).mean(),'\n',
         'bone3_Dice:', np.array(bone3_Dice).mean(),'\n',
         'bone3_Hausdorff:', np.array(bone3_Hausdorff).mean(),'\n',
         'bone4_Dice:', np.array(bone4_Dice).mean(),'\n',
         'bone4_Hausdorff:', np.array(bone4_Hausdorff).mean(),'\n',
         'whole_Dice:', np.array(whole_Dice).mean(),'\n',
         'whole_Hausdorff:', np.array(whole_Hausdorff).mean(),'\n',
         'mean_Dice:', np.array(mean_Dice).mean(),'\n',
         'mean_Hausdorff:', np.array(mean_Hausdorff).mean(),'\n',
         'weight_Dice:', np.array(weight_Dice).mean(),'\n',
         'weight_Hausdorff:', np.array(weight_Hausdorff).mean(),'\n',
        )
        assert len(names)==len(bone1_Dice)
        assert len(names)==len(bone1_Hausdorff)
        assert len(names)==len(bone2_Dice)
        assert len(names)==len(bone2_Hausdorff)
        assert len(names)==len(bone3_Dice)
        assert len(names)==len(bone3_Hausdorff)
        assert len(names)==len(bone4_Dice)
        assert len(names)==len(bone4_Hausdorff)
        assert len(names)==len(whole_Dice)
        assert len(names)==len(whole_Hausdorff)
        assert len(names)==len(mean_Dice)
        assert len(names)==len(mean_Hausdorff)
        assert len(names)==len(weight_Dice)
        assert len(names)==len(weight_Hausdorff)

        results = {'names':names,
                   'bone1_Dice': bone1_Dice,
                   'bone1_Hausdorff': bone1_Hausdorff,
                   'bone2_Dice': bone2_Dice,
                   'bone2_Hausdorff': bone2_Hausdorff,
                   'bone3_Dice': bone3_Dice,
                   'bone3_Hausdorff': bone3_Hausdorff,
                   'bone4_Dice': bone4_Dice,
                   'bone4_Hausdorff': bone4_Hausdorff,
                   'whole_Dice': whole_Dice,
                   'whole_Hausdorff':whole_Hausdorff,
                   'mean_Dice':mean_Dice,
                   'mean_Hausdorff':mean_Hausdorff,
                   'weight_Dice':weight_Dice,
                   'weight_Hausdorff':weight_Hausdorff
                   }
        results_pd = pd.DataFrame(results)
        save_csv_path = eval_reslult_pkl_path.replace('.pkl','_{}.csv'.format(dataset))
        results_pd.to_csv(save_csv_path)




