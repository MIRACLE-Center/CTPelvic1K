import numpy as np
import SimpleITK as sitk

import pickle as pkl
import matplotlib.pyplot as plt
import time
import os
import cv2

def sdf_func(segImg, name):
    """
    segImg is a sitk Image
    """
    Sdf = sitk.SignedMaurerDistanceMap(segImg, squaredDistance=False)
    # Sdf = sitk.Sigmoid(-Sdf, 50, 0, 1, 0)  # alpha, beta, max, min
    Sdf = sitk.Sigmoid(-Sdf, 2, 0, 1, 0)  # alpha, beta, max, min ### 2 for boundary weight
    seg = sitk.GetArrayFromImage(Sdf)
    # seg[seg > 0.4999] = 1  # convert sdf back to numpy array, and clip 0.5 above to 1 (inside)
    # heat_map = seg + 0.5  # putGaussianMapF(mask, sigma=50.0)
    seg = (seg-seg.min())/(seg.max()-seg.min())
    print(name, seg.shape, seg.max(), seg.min())
    return seg#heat_map

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

def load_pkl(file):
    with open(file, 'rb') as f:
        a = pkl.load(f)
    return a


def save_pkl(info, file):
    with open(file, 'wb') as f:
        pkl.dump(info, f)


def _change_label(label, idx_before, idx_after):
    label[label == idx_before] = idx_after
    return label


def _Series_dicom_reader(path):
    Reader = sitk.ImageSeriesReader()
    name = Reader.GetGDCMSeriesFileNames(path)
    Reader.SetFileNames(name)
    Image = Reader.Execute()

    image = sitk.GetArrayFromImage(Image)
    Spa = Image.GetSpacing()
    Ori = Image.GetOrigin()
    Dir = Image.GetDirection()
    return Image, image, (Spa, Ori, Dir)


def _sitk_Image_reader(path):
    Image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(Image)
    Spa = Image.GetSpacing()
    Ori = Image.GetOrigin()
    Dir = Image.GetDirection()
    return Image, image, (Spa, Ori, Dir)


def _sitk_image_writer(image, meta, path):
    Image = sitk.GetImageFromArray(image)
    if meta is None:
        pass
    else:
        Image.SetSpacing(meta[0])
        Image.SetOrigin(meta[1])
        Image.SetDirection(meta[2])
    sitk.WriteImage(Image, path)