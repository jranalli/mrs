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
DS_NAME = 'sd_finetune'


def create_dataset(data_dir, save_dir, patch_size, pad, overlap, valid_percent=0.2, visualize=False):
    # create folders and files
    patch_dir = os.path.join(save_dir, 'patches')
    misc_utils.make_dir_if_not_exist(patch_dir)
    record_file_train = open(os.path.join(save_dir, 'file_list_train.txt'), 'w+')
    record_file_valid = open(os.path.join(save_dir, 'file_list_valid.txt'), 'w+')

    files = data_utils.get_img_lbl(data_dir, 'RGB.tif', 'GT.tif')
    valid_size = int(len(files) * valid_percent)
    for cnt, (img_file, lbl_file) in enumerate(tqdm(files)):
        city_name = os.path.splitext(os.path.basename(img_file))[0].split('_')[0]
        for rgb_patch, gt_patch, y, x in data_utils.patch_tile(img_file, lbl_file, patch_size, pad, overlap):
            if visualize:
                from mrs_utils import vis_utils
                vis_utils.compare_figures([rgb_patch, gt_patch], (1, 2), fig_size=(12, 5))

            img_patchname = '{}_y{}x{}.jpg'.format(city_name, int(y), int(x))
            lbl_patchname = '{}_y{}x{}.png'.format(city_name, int(y), int(x))
            misc_utils.save_file(os.path.join(patch_dir, img_patchname), rgb_patch.astype(np.uint8))
            misc_utils.save_file(os.path.join(patch_dir, lbl_patchname), (gt_patch/255).astype(np.uint8))

            if cnt < valid_size:
                record_file_valid.write('{} {}\n'.format(img_patchname, lbl_patchname))
            else:
                record_file_train.write('{} {}\n'.format(img_patchname, lbl_patchname))


def create_dataset_3folds(data_dir, save_dir, patch_size, pad, overlap, visualize=False):
    # folds
    files = data_utils.get_img_lbl(data_dir, 'RGB.tif', 'GT.tif')
    valid_size = int(len(files) / 3)
    for fold in range(3):
        curr_dir = os.path.join(save_dir, '{}'.format(fold + 1))
        # create folders and files
        patch_dir = os.path.join(curr_dir, 'patches')
        misc_utils.make_dir_if_not_exist(patch_dir)
        record_file_train = open(os.path.join(curr_dir, 'file_list_train.txt'), 'w+')
        record_file_valid = open(os.path.join(curr_dir, 'file_list_valid.txt'), 'w+')

        for cnt, (img_file, lbl_file) in enumerate(tqdm(files)):
            city_name = os.path.splitext(os.path.basename(img_file))[0].split('_')[0]
            for rgb_patch, gt_patch, y, x in data_utils.patch_tile(img_file, lbl_file, patch_size, pad, overlap):
                if visualize:
                    from mrs_utils import vis_utils
                    vis_utils.compare_figures([rgb_patch, gt_patch], (1, 2), fig_size=(12, 5))

                img_patchname = '{}_y{}x{}.jpg'.format(city_name, int(y), int(x))
                lbl_patchname = '{}_y{}x{}.png'.format(city_name, int(y), int(x))
                misc_utils.save_file(os.path.join(patch_dir, img_patchname), rgb_patch.astype(np.uint8))
                misc_utils.save_file(os.path.join(patch_dir, lbl_patchname), (gt_patch/255).astype(np.uint8))

                if fold == 0:
                    if cnt < valid_size:
                        record_file_valid.write('{} {}\n'.format(img_patchname, lbl_patchname))
                    else:
                        record_file_train.write('{} {}\n'.format(img_patchname, lbl_patchname))
                elif fold == 1:
                    if valid_size <= cnt < valid_size*2:
                        record_file_valid.write('{} {}\n'.format(img_patchname, lbl_patchname))
                    else:
                        record_file_train.write('{} {}\n'.format(img_patchname, lbl_patchname))
                else:
                    if cnt >= valid_size*2:
                        record_file_valid.write('{} {}\n'.format(img_patchname, lbl_patchname))
                    else:
                        record_file_train.write('{} {}\n'.format(img_patchname, lbl_patchname))


def get_images(data_dir, valid_percent=0.2):
    files = data_utils.get_img_lbl(data_dir, 'RGB.tif', 'GT.tif')
    valid_size = int(len(files) * valid_percent)
    rgb_files, gt_files = [], []
    for cnt, (img_file, lbl_file) in enumerate(files):
        if cnt < valid_size:
            rgb_files.append(img_file)
            gt_files.append(lbl_file)
    return rgb_files, gt_files


def get_stats(img_dir):
    from data import data_utils
    from glob import glob
    rgb_imgs = glob(os.path.join(img_dir, '*.jpg'))
    ds_mean, ds_std = data_utils.get_ds_stats(rgb_imgs)
    return np.stack([ds_mean, ds_std], axis=0)


def get_stats_pb(img_dir):
    val = process_block.ValueComputeProcess(DS_NAME, os.path.join(os.path.dirname(__file__), '../stats/builtin'),
                                            os.path.join(os.path.dirname(__file__), '../stats/builtin/{}.npy'.format(DS_NAME)), func=get_stats).\
        run(img_dir=img_dir).val
    val_test = val
    return val, val_test


if __name__ == '__main__':
    ps = 512
    ol = 0
    pd = 0
    create_dataset(data_dir=r'/home/wh145/data/SDhist/tif_files/',
                   save_dir=r'/home/wh145/data/SDhist/ps_512', patch_size=(ps, ps), pad=pd, overlap=ol, visualize=False)
