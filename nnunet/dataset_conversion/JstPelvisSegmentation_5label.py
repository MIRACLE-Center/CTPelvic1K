from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from multiprocessing import Pool
import numpy as np
from scipy.ndimage import label
import shutil
import argparse


def export_segmentations(indir, outdir):
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        sitk.WriteImage(img, outfname)


def export_segmentations_postprocess(indir, outdir):
    maybe_mkdir_p(outdir)
    niftis = subfiles(indir, suffix='nii.gz', join=False)
    for n in niftis:
        print("\n", n)
        identifier = str(n.split("_")[-1][:-7])
        outfname = join(outdir, "test-segmentation-%s.nii" % identifier)
        img = sitk.ReadImage(join(indir, n))
        img_npy = sitk.GetArrayFromImage(img)
        lmap, num_objects = label((img_npy > 0).astype(int))
        sizes = []
        for o in range(1, num_objects + 1):
            sizes.append((lmap == o).sum())
        mx = np.argmax(sizes) + 1
        print(sizes)
        img_npy[lmap != mx] = 0
        img_new = sitk.GetImageFromArray(img_npy)
        img_new.CopyInformation(img)
        sitk.WriteImage(img_new, outfname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default=None,
                        help="path to dataset CTPelvic1K")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="random seed (default: 42)")
    opts = parser.parse_args()
    """
        dataset format:
            image: *_data.nii.gz
            label: *_mask_4label.nii.gz
        test: 
            image: *_data.nii.gz
        Data will be converted into a unified format used in nnunet
    """
    train_dir = opts.train_dir
    output_folder = opts.output_dir
    test_dir = "/path/to/testing dataset"

    img_dir = join(output_folder, "imagesTr")
    lab_dir = join(output_folder, "labelsTr")
    img_dir_te = join(output_folder, "imagesTs")
    maybe_mkdir_p(img_dir)
    maybe_mkdir_p(lab_dir)
    maybe_mkdir_p(img_dir_te)


    def load_save_train(args):
        data_file, seg_file = args
        pat_id = data_file.split("/")[-1]
        pat_id = "train_" + pat_id.split("-")[-1][:-12]

        shutil.copy(data_file, join(img_dir, pat_id + ".nii.gz"))

        shutil.copy(seg_file, join(lab_dir, pat_id + ".nii.gz"))
        return pat_id
    def load_save_test(args):
        data_file = args
        pat_id = data_file.split("/")[-1]
        pat_id = "test_" + pat_id.split("-")[-1][:-12]

        shutil.copy(data_file, join(img_dir_te, pat_id + ".nii.gz"))
        return pat_id


    nii_files_tr_data = subfiles(train_dir, True, None, "_data.nii.gz", True)
    nii_files_tr_seg  = subfiles(train_dir, True, None, "_mask_4label.nii.gz", True)

    nii_files_ts      = subfiles(test_dir, True, None, "_data.nii.gz", True)

    p = Pool(16)
    train_ids = p.map(load_save_train, zip(nii_files_tr_data, nii_files_tr_seg))
    test_ids = p.map(load_save_test, nii_files_ts)
    p.close()
    p.join()

    json_dict = OrderedDict()
    json_dict['name'] = "CTPelvic1K_4label"
    json_dict['description'] = "CTPelvic1K_4label"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.1"
    json_dict['modality'] = {
        "0": "CT"
    }

    json_dict['labels'] = {
        "0": "background",
        "1": "sacrum",
        "2": "right_hip",
        "3": "left_hip",
        "4": "lumbar_vertebra"
    }

    json_dict['numTraining'] = len(train_ids)

    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i,
                              "label": "./labelsTr/%s.nii.gz" % i} for i in train_ids]

    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)
