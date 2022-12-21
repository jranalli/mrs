"""

"""


# Built-in
import os
from glob import glob

# Libs
import numpy as np
from tqdm import tqdm
from natsort import natsorted

# Own modules
from data import data_utils
from mrs_utils import misc_utils, process_block

# Settings
DS_NAME = 'nyc_pv'


def patch_tile(rgb_file, gt_file, patch_size, pad, overlap):
    """
    Extract the given rgb and gt tiles into patches
    :param rgb_file: path to the rgb file
    :param gt_file: path to the gt file
    :param patch_size: size of the patches, should be a tuple of (h, w)
    :param pad: #pixels to be padded around each tile, should be either one element or four elements
    :param overlap: #overlapping pixels between two patches in both vertical and horizontal direction
    :return: rgb and gt patches as well as coordinates
    """
    rgb = misc_utils.load_file(rgb_file)
    gt = misc_utils.load_file(gt_file)
    np.testing.assert_array_equal(rgb.shape[:2], gt.shape)
    grid_list = data_utils.make_grid(
        np.array(rgb.shape[:2]) + 2 * pad, patch_size, overlap)
    if pad > 0:
        rgb = data_utils.pad_image(rgb, pad)
        gt = data_utils.pad_image(gt, pad)
    for y, x in grid_list:
        rgb_patch = data_utils.crop_image(
            rgb, y, x, patch_size[0], patch_size[1])
        gt_patch = data_utils.crop_image(
            gt, y, x, patch_size[0], patch_size[1])
        yield rgb_patch, gt_patch, y, x


# Get names of all files, split test and validation
def split_sets(img_files, lbl_files, test_frac=0.2, train_frac=0.72, valid_frac=0.08):

    cum_test_frac = test_frac
    cum_train_frac = cum_test_frac + train_frac
    cum_valid_frac = cum_train_frac + valid_frac

    test_files = []
    train_files = []
    valid_files = []
    for i, pair in enumerate(zip(img_files, lbl_files)):
        for dataset, frac in zip([test_files, train_files, valid_files], [cum_test_frac, cum_train_frac, cum_valid_frac]):
            if i <= int(frac * len(img_files)):
                dataset.append(pair)
                break
    return (test_files, train_files, valid_files)


# Split train & valid images to patches and save to file
def create_dataset(img_dir, lbl_dir, save_dir, patch_size, pad, overlap, min_size=50):
    # create folders and file lists
    misc_utils.make_dir_if_not_exist(save_dir)
    patch_dir = os.path.join(save_dir, 'patches')
    misc_utils.make_dir_if_not_exist(patch_dir)


    img_files = natsorted(glob(os.path.join(img_dir, '*.png')))
    lbl_files = natsorted(glob(os.path.join(lbl_dir, '*.png')))

    rgbs = []
    lbls = []
    for i, pair in enumerate(zip(img_files, lbl_files)):
        basename = os.path.splitext(os.path.basename(pair[0]))[0]
        for rgb_patch, gt_patch, y, x in patch_tile(pair[0], pair[1], patch_size, pad, overlap):
            if np.sum(gt_patch) > min_size:
                rgb_patchname = '{}_y{}x{}.png'.format(basename, int(y), int(x))
                gt_patchname = '{}_y{}x{}_lbl.png'.format(basename, int(y), int(x))
                rgbs.append(rgb_patchname)
                lbls.append(gt_patchname)
                misc_utils.save_file(os.path.join(
                    patch_dir, rgb_patchname), rgb_patch.astype(np.uint8))
                misc_utils.save_file(os.path.join(
                    patch_dir, gt_patchname), (gt_patch).astype(np.uint8))
            else:
                # this patch is blank
                pass

    record_file_train = open(os.path.join(
        save_dir, 'file_list_train.txt'), 'w+')
    record_file_valid = open(os.path.join(
        save_dir, 'file_list_valid.txt'), 'w+')
    record_file_test = open(os.path.join(
        save_dir, 'file_list_test.txt'), 'w+')

    test_files, train_files, valid_files = split_sets(rgbs, lbls)

    for img_file, lbl_file in test_files:
        record_file_test.write('{} {}\n'.format(img_file, lbl_file))
    for img_file, lbl_file in train_files:
        record_file_train.write('{} {}\n'.format(img_file, lbl_file))
    for img_file, lbl_file in valid_files:
        record_file_valid.write('{} {}\n'.format(img_file, lbl_file))


def get_stats(img_dir):
    from data import data_utils
    from glob import glob
    rgb_imgs = []
    rgb_imgs.extend(glob(os.path.join(img_dir, '*.png')))
    ds_mean, ds_std = data_utils.get_ds_stats(rgb_imgs)
    print('Mean: {}'.format(ds_mean))
    print('Std: {}'.format(ds_std))
    return np.stack([ds_mean, ds_std], axis=0)


if __name__ == '__main__':

    create_dataset(img_dir=r'f:\solardnn\NYC_mrs\img',
                   lbl_dir=r'f:\solardnn\NYC_mrs\mask',
                   save_dir=r'f:\solardnn\NYC_mrs',
                   patch_size=(512, 512),
                   pad=0,
                   overlap=0,
                   min_size=50)

    get_stats('f:\\solardnn\\NYC_mrs\\img')