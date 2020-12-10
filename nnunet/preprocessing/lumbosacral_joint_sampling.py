"""
make a dataset for lumbosacral joint prediction.
three class: background, sacrum, lumbar - 0,1,2

"""
import numpy as np
from multiprocessing import Pool
from functools import partial
import cv2
import  os
import  pickle as pkl

def get_reasonable_crops_of_lumbar(gt3d, patch_size, stage):
    d, h, w = gt3d.shape
    print((d,h,w))

    lambar = np.where(gt3d == 4, 1, 0)
    sacral = np.where(gt3d == 1, 1, 0)
    upside_down = False
    d_lambar = np.argwhere(lambar > 0)[:, 0]
    d_sacral = np.argwhere(sacral > 0)[:, 0]
    if d_sacral.mean() > d_lambar.mean():
        upside_down = True
        print(upside_down)

    if lambar.sum()==0:
        if stage==0:
            c_s = 50
        elif stage==1:
            c_s = 100
        if upside_down:
            return [(0, int(patch_size[0])),
                    (int(h//2-c_s), int(h//2+c_s)),
                    (int(w//2-c_s), int(w//2+c_s))]
        else:
            return [(int(d-patch_size[0]), int(d-1)),
                    (int(h // 2 - c_s), int(h // 2 + c_s)),
                    (int(w // 2 - c_s), int(w // 2 + c_s))]

    if upside_down:
        lambar_boundary_d = sorted(set(d_lambar))[-5]+4
    else:
        lambar_boundary_d = sorted(set(d_lambar))[4]-4

    h_lambar = np.argwhere(lambar > 0)[:, 1]
    w_lambar = np.argwhere(lambar > 0)[:, 2]
    lambar_center_h = (max(h_lambar)+min(h_lambar))//2
    lambar_center_w = (max(w_lambar)+min(w_lambar))//2
    print(max(h_lambar), min(h_lambar))
    print(max(w_lambar), min(w_lambar))
    print(lambar_boundary_d, lambar_center_h, lambar_center_w)

    valid_crop = [(int(max(0, lambar_boundary_d-patch_size[0]//2)), int(min(d, lambar_boundary_d + patch_size[0]//2))),
                  (int(max(0, lambar_center_h-patch_size[1]//2)), int(min(h, lambar_center_h + patch_size[1]//2))),
                  (int(max(0, lambar_center_w-patch_size[2]//2)), int(min(w, lambar_center_w + patch_size[2]//2)))]

    return valid_crop

def _main_3d_one_case(name, path, crop_size, stage, img_save_path):
    print(name)
    try:
        array = np.load(path + '/' + name + '.npy')
    except Exception as _:
        array = np.load(path + '/' + name + '.npz')['data']

    with open(path + '/' + name + '.pkl', 'rb') as f:
        info = pkl.load(f)

    reasonal = get_reasonable_crops_of_lumbar(array[-1, :, :, :], patch_size=crop_size, stage=stage)
    info['Lumbosacral_Region'] = reasonal

    # backup raw pkl file
    os.rename(path + '/' + name + '.pkl', path + '/' + name + '_backup.pkl')
    # save new pkl file
    with open(path + '/' + name + '.pkl', 'wb') as f:
        pkl.dump(info, f)
    print(reasonal)

    # check reasonal
    saveimg = array[0][(reasonal[0][0] + reasonal[0][1]) // 2, reasonal[1][0]:reasonal[1][1],
              reasonal[2][0]:reasonal[2][1]]
    saveimg = (saveimg - saveimg.min()) / (saveimg.max() - saveimg.min()) * 255
    cv2.imwrite(img_save_path + '/' + name + '.png', saveimg)
def main_3d(base_path, check_save_path):
    base_path = base_path
    stage = 1

    path = f'{base_path}/nnUNet_stage{stage}'
    plans_path = f'{base_path}/nnUNetPlans_plans_3D.pkl'

    names = os.listdir(path)
    names = [i[:-4] for i in names if i.endswith('.npy')]

    print(len(names),'files to process...')

    with open(plans_path, "rb") as f:
        plans_info = pkl.load(f)

    crop_size = plans_info["plans_per_stage"][stage]["patch_size"]  # array([ 96, 160, 128])
    crop_size[2] = 180 # enlarge

    assert stage==1

    img_save_path = check_save_path
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    pool = Pool()
    func = partial(_main_3d_one_case, path=path, crop_size=crop_size, stage=stage, img_save_path=img_save_path)
    _ = pool.map(func, names)
    pool.close()
    pool.join()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_path", type=str, default=None,
                        help="path to dataset CTPelvic1K")
    parser.add_argument("--check_save_path", type=str, default=None,
                        help="random seed (default: 42)")
    opts = parser.parse_args()

    main_3d(base_path=opts.processed_path,
            check_save_path=opts.check_save_path)

