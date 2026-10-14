"""Validate U-Net cell (biomass) segmentation models on EXP-26-CE8718 microchamber stacks.

Each model is run exactly as in the TB_analysis_Mjolnir_S2 pipeline (its `_segment_unet`), and the
masks are compared per frame with a reference:
  REF/Pos201   -> manually corrected masks (ground truth)
  REF_2/Pos201 -> masks of the old-parameter z-score model (proxy reference, no manual correction)

Usage:
  python validate_microchambers.py MODEL[:zscore|p99] [MODEL[:norm] ...] [--out DIR] [--min-size 30]

  The normalization must match the one the model was trained with. If it is not given, the
  checkpoint's 'normalize' entry is used when present, otherwise zscore.

Example:
  python validate_microchambers.py \
      ../../models/unetTB_256_261008_microchambers_old_param.pth:zscore \
      ../../models/unetTB_256_261009_microchambers_old_param_custom_normalize99.pth:p99 --out /tmp/val
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from skimage import io

PIPELINE_DIR = '/home/spartak/elflab/BSL3/analysis/EXP-26-CB9775/TB_analysis_Mjolnir_S2'
sys.path.insert(0, PIPELINE_DIR)
from tbseg.segment_stack import _segment_unet, custom_normalize, custom_normalize99  # noqa: E402

from utils.unet import UNet  # noqa: E402

OUT_ROOT = '/home/spartak/elflab/BSL3/analysis/EXP-26-CB9775/output_S2/EXP-26-CE8718/unet_old_params'
DATASETS = {
    'REF/Pos201': {'images': f'{OUT_ROOT}/REF/Pos201/chamber_location0',
                   'reference': f'{OUT_ROOT}/REF/Pos201/masks_location0_true',
                   'ref_kind': 'manual'},
    'REF_2/Pos201': {'images': f'{OUT_ROOT}/REF_2/Pos201/chamber_location0',
                     'reference': f'{OUT_ROOT}/REF_2/Pos201/masks_location0',
                     'ref_kind': 'old model'},
}
NORMALIZE = {'zscore': custom_normalize, 'p99': custom_normalize99}


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = ckpt['model_state_dict']
    # max_filters is the widest layer of the network
    max_filters = max(v.shape[0] for v in state.values() if v.dim() > 0)
    net = UNet(max_filters=max_filters)
    net.load_state_dict(state)
    net.eval().cuda()
    return net, max_filters, ckpt.get('normalize')


def iou_dice(pred, ref):
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    total = pred.sum() + ref.sum()
    return (inter / union if union else 1.0), (2 * inter / total if total else 1.0)


def evaluate(net, normalize, dataset, min_size):
    images = sorted(glob.glob(os.path.join(dataset['images'], '*.tif')))
    refs = sorted(glob.glob(os.path.join(dataset['reference'], '*.tif')))
    assert len(images) == len(refs) > 0, f"{len(images)} images vs {len(refs)} reference masks"
    rows = []
    with torch.no_grad():
        for f_im, f_ref in zip(images, refs):
            im = io.imread(f_im).astype(np.float32)
            ref = io.imread(f_ref) > 0
            pred = _segment_unet(im, net, min_size=min_size, normalize=normalize) > 0
            iou, dice = iou_dice(pred, ref)
            rows.append((iou, dice, pred.sum(), ref.sum()))
    return np.array(rows, dtype=float), images


def summarize(rows):
    iou, dice, a_pred, a_ref = rows.T
    n = min(10, len(iou))
    # growth over the stack: log area ratio last/first frame (what the growth-rate analysis depends on)
    growth = lambda a: np.log(max(a[-1], 1) / max(a[0], 1))
    return {
        'IoU': iou.mean(), 'IoU_first10': iou[:n].mean(), 'IoU_last10': iou[-n:].mean(),
        'Dice': dice.mean(),
        'area_ratio': np.median(a_pred / np.maximum(a_ref, 1)),
        'log_growth_pred': growth(a_pred), 'log_growth_ref': growth(a_ref),
    }


def save_outputs(out_dir, label, ds_name, rows, images, net, normalize, min_size):
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{label}__{ds_name.replace('/', '_')}"
    np.savetxt(os.path.join(out_dir, tag + '.csv'), rows, delimiter=',',
               header='iou,dice,area_pred,area_ref', comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    refs = sorted(glob.glob(os.path.join(DATASETS[ds_name]['reference'], '*.tif')))
    frames = [0, len(images) // 2, len(images) - 1]
    fig, axs = plt.subplots(len(frames) + 1, 1, figsize=(8, 3 * (len(frames) + 1)))
    for ax, k in zip(axs, frames):
        im = io.imread(images[k]).astype(np.float32)
        with torch.no_grad():
            pred = _segment_unet(im, net, min_size=min_size, normalize=normalize) > 0
        ref = io.imread(refs[k]) > 0
        ax.imshow(im, cmap='gray')
        ax.contour(ref, levels=[0.5], colors='lime', linewidths=0.8)
        ax.contour(pred, levels=[0.5], colors='red', linewidths=0.8)
        ax.set_title(f'frame {k}: IoU {rows[k, 0]:.2f}  (green reference, red prediction)', fontsize=9)
        ax.axis('off')
    axs[-1].plot(rows[:, 3], 'g-', label='reference')
    axs[-1].plot(rows[:, 2], 'r-', label='prediction')
    axs[-1].set_xlabel('frame')
    axs[-1].set_ylabel('area (px)')
    axs[-1].legend()
    fig.suptitle(tag, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, tag + '.png'), dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('models', nargs='+', help='model.pth[:zscore|p99]')
    parser.add_argument('--out', help='directory for per-frame CSVs and overlay/area plots')
    parser.add_argument('--min-size', type=int, default=30, help='remove objects smaller than this (px)')
    args = parser.parse_args()

    results = []
    for spec in args.models:
        path, _, norm = spec.partition(':')
        net, max_filters, ckpt_norm = load_model(path)
        norm = norm or ckpt_norm or 'zscore'
        label = os.path.splitext(os.path.basename(path))[0]
        for ds_name, ds in DATASETS.items():
            rows, images = evaluate(net, NORMALIZE[norm], ds, args.min_size)
            results.append((label, max_filters, norm, ds_name, ds['ref_kind'], summarize(rows)))
            if args.out:
                save_outputs(args.out, label, ds_name, rows, images, net, NORMALIZE[norm], args.min_size)
        del net
        torch.cuda.empty_cache()

    header = (f"{'model':60s} {'filt':>4s} {'norm':>6s} {'dataset':13s} {'reference':9s} "
              f"{'IoU':>5s} {'first10':>7s} {'last10':>6s} {'Dice':>5s} {'area':>5s} {'lnGrowth':>14s}")
    print(header)
    print('-' * len(header))
    for label, mf, norm, ds_name, ref_kind, s in results:
        print(f"{label[:60]:60s} {mf:4d} {norm:>6s} {ds_name:13s} {ref_kind:9s} "
              f"{s['IoU']:5.3f} {s['IoU_first10']:7.3f} {s['IoU_last10']:6.3f} {s['Dice']:5.3f} "
              f"{s['area_ratio']:5.2f} {s['log_growth_pred']:6.2f}/{s['log_growth_ref']:<6.2f}")
    print("\narea = median(predicted/reference area); lnGrowth = ln(area last/first frame), prediction/reference")


if __name__ == '__main__':
    main()
