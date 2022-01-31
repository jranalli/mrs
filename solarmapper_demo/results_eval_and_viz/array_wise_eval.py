"""
Perform array-wise performance evaluation and plot curves.

Supported types of curves:
    - Precision-recall curves
    - ROC curve (false positive rate per unit are on the x-axis)
"""

import os
import argparse

from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import numpy as np
from skimage import io, measure
from sklearn import metrics

from mrs_utils import eval_utils


# File path related
CONF_DIR = '/home/wh145/results/solarmapper/ca_only/ec_1e-3_dc_1e-2/cls_wt_1_1/old_gt_sept2021/ecresnet50_dcunet_dsct_new_non_random_3_splits_lre1e-03_lrd1e-02_ep180_bs7_ds30_dr0p1_crxent7p0_softiou3p0'
GT_DIR = '/home/wh145/data/ct_new/random_2_1'
OUTPUT_DIR = '/home/wh145/mrs/tasks/final_eval/ct_test'

# Post-processing parameters
MIN_SIZE = 10 # Minimum panel polygon size threshold
MIN_TH = 128 # Minimum confidence threshold
IOU_TH = 0.5 # IoU threshold for array-wise matching
DILATION_SIZE = 10 # Dilation size for arrayt-wise grouping


def plain_post_proc(conf, min_conf, min_size):
    # Polygon size and confidence thresholding
    tmp = conf >= min_conf
    label = measure.label(tmp)
    props = measure.regionprops(label, conf)

    dummy = np.zeros(conf.shape)
    for p in props:
        if p.area > min_size:
            for x, y in p.coords:
                dummy[x, y] = 1
    return dummy


def get_tile_name_list(gt_dir):

    tile_name_list = [t.split('.')[0] \
        for t in os.listdir(gt_dir) if t.endswith('.jpg')]
    
    return tile_name_list


def read_gt(gt_dir, tile_name_list):

    gt_dict = dict(
        zip(
            tile_name_list, 
            [io.imread(os.path.join(gt_dir, f+'.png')) for f in tile_name_list]
            )
        )

    return gt_dict


def read_conf(conf_dir, tile_name_list):

    conf_dict = dict(
        zip(
            tile_name_list,
            [io.imread(os.path.join(conf_dir, f+'_conf.png')) / 255
                for f in tile_name_list]
            )
        )

    return conf_dict


def make_fig_fname(params_dict, fname_note=None):
    params_name = 'object_pr_min_th_{}_dilation_size_{}px_iou_th_{}'.format(
        params_dict['min_th'], params_dict['dilation_size'], str(params_dict['iou_th']).replace('.', 'p')
    )

    if fname_note:
        return f'{params_name}_{fname_note}.png'
    else:
        return params_name + '.png'


def make_title(params_dict):
    return '\n'.join([
                f'Minimum confidence threshold: {params_dict["min_th"]}',
                f'Dilation size: {params_dict["dilation_size"] * 0.3:.1f}$m$',
                f'IoU threshold: {params_dict["iou_th"]}'
            ])


def plot_object_curve(
    tile_name_list, conf_dict, gt_dict, params_dict, return_values=False, output_dir=None, 
    curve_type='roc', show_title=False, savefig=False, curve_label=None, tile_set=False):

    if savefig:
        plt.figure(figsize=(8, 8))
    
    # Read confidence maps and corresponding ground truth masks
    conf_list, true_list = [], []
    for tile_name in tqdm(tile_name_list, desc='Tiles'):
        conf_img, lbl_img = conf_dict[tile_name], gt_dict[tile_name][:, :]
        conf_tile, true_tile = eval_utils.score(
            conf_img, 
            lbl_img, 
            min_region=params_dict['min_size'], 
            min_th=params_dict['min_th'] / 255, 
            dilation_size=params_dict['dilation_size'], 
            iou_th=params_dict['iou_th']
            )
        conf_list.extend(conf_tile)
        true_list.extend(true_tile)

    if curve_type == 'pr':
    
        ap, p, r, th = eval_utils.get_precision_recall(conf_list, true_list)

        if tile_set:
            with open(os.path.join(output_dir, f'{tile_set}_pr_results.txt'), 'w') as f:
                f.write('precision, recall, threshold\n')
                for trio in zip(p, r, th):
                    f.write(','.join([str(v) for v in list(trio)]))
                    f.write('\n')

        if curve_label:
            plt.plot(r[1:], p[1:], label=f'{curve_label}: AP={ap:.4f}')
        else:
            plt.plot(r[1:], p[1:], label=f'AP={ap:.4f}')

        if savefig:
            if show_title:
                plt.title(make_title(params_dict))

            plt.xlim([0, 1.05])
            plt.ylim([0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend(fontsize=18)
            plt.tight_layout()

            save_path = os.path.join(
                output_dir, make_fig_fname(params_dict))
            plt.savefig(save_path, dpi=300)
            plt.close()
    
        if return_values:
            return ap, p, r
        else:
            pass
    
    elif curve_type == 'roc':
        _, _, r, th = eval_utils.get_precision_recall(conf_list, true_list)
        fps = eval_utils.get_fps(conf_list, true_list)
        total_area = ((0.001 * 2541 * 0.3) ** 2) * len(tile_name_list) # km^2
        fps_per_unit_area = fps / total_area
        auc = metrics.auc(fps_per_unit_area, r[1:]) / (np.max(fps_per_unit_area))

        if tile_set:
            with open(os.path.join(output_dir, f'{tile_set}_roc_results.txt'), 'w') as f:
                f.write('fp_rate, recall, threshold\n')
                for trio in zip(fps_per_unit_area, r, th):
                    f.write(','.join([str(v) for v in list(trio)]))
                    f.write('\n')

        if curve_label:
            plt.plot(fps_per_unit_area, r[1:], label=f'{curve_label}, AUC={auc:.4f}')
        else:
            plt.plot(fps_per_unit_area, r[1:], label=f'AUC={auc:.4f}')

        if savefig:
            if show_title:
                plt.title(make_title(params_dict))

            plt.ylim([0, 1.05])
            plt.xlabel('False positive per $m^2$')
            plt.ylabel('True positive rate')
            plt.legend(fontsize=18)
            plt.tight_layout()

            save_path = os.path.join(
                output_dir, make_fig_fname(params_dict))
            plt.savefig(save_path, dpi=300)
            plt.close()

        if return_values:
            return auc, fps_per_unit_area, r
        else:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_dir', type=str, default=CONF_DIR)
    parser.add_argument('--gt_dir', type=str, default=GT_DIR)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    parser.add_argument('--min_size', type=float, default=MIN_SIZE)
    parser.add_argument('--min_th', type=float, default=MIN_TH)
    parser.add_argument('--iou_th', type=float, default=IOU_TH)
    parser.add_argument('--dilation_size', type=float, default=DILATION_SIZE)
    parser.add_argument('--fname_note', type=str, default='CT_new')
    parser.add_argument('--curve_type', type=str, default='pr')

    super_args = parser.parse_args()
    
    if not os.path.exists(super_args.output_dir):
        os.makedirs(super_args.output_dir)

    params_dict = {
        'min_size': super_args.min_size,
        'min_th': super_args.min_th,
        'iou_th': super_args.iou_th,
        'dilation_size': super_args.dilation_size,
    }

    plt.figure(figsize=(8, 8))

    for tile_set in tqdm(['commercial', 'all', 'residential']):

        tile_name_list = get_tile_name_list(gt_dir=super_args.gt_dir, tile_set=tile_set)
        conf_dict = read_conf(conf_dir=super_args.conf_dir, tile_name_list=tile_name_list)
        gt_dict = read_gt(gt_dir=super_args.gt_dir, tile_name_list=tile_name_list)

        plot_object_curve(
            tile_name_list=tile_name_list,
            conf_dict=conf_dict,
            gt_dict=gt_dict,
            output_dir=super_args.output_dir,
            params_dict=params_dict,
            curve_label=tile_set.capitalize(),
            tile_set=tile_set
        )
    
    if super_args.curve_type == 'pr':
        plt.xlim([0, 1.05])
        plt.ylim([0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
    elif super_args.curve_type == 'roc':
        plt.xlim(left=0)
        plt.ylim([0, 1.05])
        plt.xlabel('False positive per $km^2$')
        plt.ylabel('True positive rate')

    plt.legend(fontsize=18)
    plt.tight_layout()
    save_path = os.path.join(
        super_args.output_dir, make_fig_fname(params_dict, fname_note=super_args.fname_note))
    plt.savefig(save_path, dpi=300)
    plt.close()
