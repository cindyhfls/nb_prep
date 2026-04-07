#!/usr/bin/env python
# coding: utf-8

from nilearn.image import new_img_like,load_img,resample_to_img
import os
from glob import glob
import numpy as np
import sys

aseg_mapping = {
    3:  'l-cerebralCtx',
    42: 'r-cerebralCtx',

    8:  'l-cerebellum',
    47: 'r-cerebellum',
    16: 'brain-stem',

    10: 'l-thalamus',
    49: 'r-thalamus',
    11: 'l-caudate',
    50: 'r-caudate',
    12: 'l-putamen',
    51: 'r-putamen',
    13: 'l-pallidum',
    52: 'r-pallidum',
    17: 'l-hippocampus',
    53: 'r-hippocampus',
    18: 'l-amygdala',
    54: 'r-amygdala',
    26: 'l-accumbens',
    58: 'r-accumbens',
    28: 'l-ventral-diencephalon',
    60: 'r-ventral-diencephalon',
}

fmriprep_out_root = '/dartfs/rc/lab/H/HaxbyLab/feilong/fmriprep_out_root/'
out_dir = "/dartfs/rc/lab/H/HaxbyLab/datasets/spacetop/24.1.1/resampled/"
tag = "1step_fmriprep_overlap"
space = "mni-2mm"

mask_file = load_img('shared_utils/support_files/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_seg-aseg_dseg.nii.gz')
# mask_file: https://github.com/cindyhfls/nb_prep/blob/24f3e0206ca2d42e50b71ea6872b34de1ce5cdbb/src/nb_prep/atlas/tpl-MNI152NLin2009cAsym_res-02_atlas-aseg_dseg.nii.gz
atlas = mask_file.get_fdata()
dset = nb.SpaceTop(
        fp_version="24.1.1"
    )
sids = dset.subject_sets['movie13']

# for subj_id in range(5,10):
subj_id = int(sys.argv[1])
img_fns = sorted(glob(os.path.join(fmriprep_out_root, 'spacetop_24.1.1', f'output_{sids[subj_id]}',f'sub-{ssids[subj_id]}','ses-*','func', '*alignvideo*preproc_bold.nii.gz')))
for curr_fn in img_fns:
data_img = load_img(curr_fn)
resampled_data_img = resample_to_img(data_img,mask_file)
resampled = resampled_data_img.get_fdata()
output = {}
for idx, name in aseg_mapping.items():
    if idx in [3, 42, 8, 47, 16] :
        continue
    mask = (atlas == idx)
    out_fn = f'{out_dir}/{space}/{name}/{tag}/{os.path.basename(curr_fn).replace("_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",".npy")}'
    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    np.save(out_fn, resampled[mask].T)
del resampled
