import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
import json
import warnings

from .regression import read_nuisance_regressors, legendre_regressors

# Make temporal mask
def compute_temporal_mask(in_fn,conf,confounds_dir,fd_threshold=None,std_dvars_threshold=None,min_contiguous=None):
    out_fn = in_fn.replace('_desc-confounds_', '_desc-mask_')[:-4] + '.npy'
    out_fn = os.path.join(confounds_dir, os.path.basename(out_fn))
    mask = np.zeros((conf.shape[0], ), dtype=bool)
    if fd_threshold is not None or std_dvars_threshold is not None:
        if fd_threshold is not None:
            mask = np.logical_or(mask,conf['framewise_displacement']>fd_threshold)
        if std_dvars_threshold is not None:
            mask = np.logical_or(mask,conf['std_dvars']>std_dvars_threshold)
        non_steady_state_counter = 0
        for key in conf.columns:
            if key.startswith('non_steady_state_outlier'):
                mask = np.logical_or(mask, conf[key].values)
                non_steady_state_counter += 1
        
        mask = np.logical_not(mask) # True = Keep, False = Outlier
        if min_contiguous:
            result = mask.copy()
            n = len(mask)
            i = 0
            while i < n:
                if mask[i]:  # Found a True value
                    j = i
                    while j < n and mask[j]:
                        j += 1
                    if (j - i) < min_contiguous:
                        result[i:j] = False
                    i = j  # Skip to the end of this segment
                else:
                    i += 1
            mask = result
        # save the dictionary
        mask_descriptions = {
            "fd_threshold": fd_threshold,
            "std_dvars_threshold": std_dvars_threshold,
            "non_steady_state_counter":non_steady_state_counter,
            "contiguous_frames":min_contiguous
        }
        with open(out_fn.replace(".npy",".json"), "w") as f:
            json.dump(mask_descriptions, f, indent=4)
    else:
        warnings.warn("Saving fmriprep default motion outliers only",UserWarning)
        # LEGACY
        # Use the default fmriprep output (defaults are FD > 0.5 mm or standardized DVARS > 1.5 if command line options --fd-spike-threshold and --dvars-spike-threshold were not set)
        # I would not recommend this because this is so obscure and it is hard to tell whether the user had set those arguments but that's what Feilong did, so keeping it here for backward compatibility
        # Also note that this did not exclude non-steady state volumes, I am keeping it as is
        for key in conf.columns:
            if key.startswith('motion_outlier'):
                mask = np.logical_or(mask, conf[key].values)
        mask = np.logical_not(mask)
        mask_descriptions = {
            "description":"legacy mask with fmriprep default motion outlier only (no non-steady-state outlier)",
            "fd_threshold":"see fmriprep",
            "std_dvars_threshold": "see fmriprep",
            "non_steady_state_counter": "not used",
            "contiguous_frames":1
        }
        with open(out_fn.replace(".npy",".json"), "w") as f:
            json.dump(mask_descriptions, f, indent=4)
    np.save(out_fn, mask)

def compute_regressors(in_fn,conf,confounds_dir,columns=None):
    out_fn = in_fn.replace(".tsv",".npy")
    out_fn = os.path.join(confounds_dir, os.path.basename(out_fn))
    if not columns:
        # The default columns were originally buried in another script, I want to take it out for transparency
        warnings.warn("Legacy confound regressors were saved, should have included cosine regressors and drop framewise displacement?",UserWarning)
        columns = \
        ['a_comp_cor_%02d' % (_, ) for _ in range(6)] + \
        ['framewise_displacement'] + \
        ['trans_x', 'trans_y', 'trans_z'] + \
        ['rot_x', 'rot_y', 'rot_z'] + \
        ['trans_x_derivative1', 'trans_y_derivative1', 'trans_z_derivative1'] + \
        ['rot_x_derivative1', 'rot_y_derivative1', 'rot_z_derivative1']
    regressors = np.nan_to_num(conf[columns])
    regressors = np.hstack([
        regressors,
        legendre_regressors(polyord=2, n_tp=regressors.shape[0])])
    np.save(out_fn, regressors)

def save_basic_confounds(in_fn,conf,confounds_dir):
    out_fn = os.path.join(confounds_dir, os.path.basename(in_fn))
    # Make a copy of the confounds *.tsv Update 2025.09.04 only copy the very basic sets inspired by the xcp-d options
    all_columns = \
        ['framewise_displacement','std_dvars'] + \
        ['a_comp_cor_%02d' % (_, ) for _ in range(6)] + \
        ['c_comp_cor_%02d' % (_, ) for _ in range(5)] + \
        ['w_comp_cor_%02d' % (_, ) for _ in range(5)] + \
        ['global_signal','csf','white_matter']+\
        ['trans_x', 'trans_y', 'trans_z'] + \
        ['rot_x', 'rot_y', 'rot_z'] 
    mask = conf.columns.str.contains(r'(cosine)')
    # mask = conf.columns.str.contains(r'(trans_|rot_|global_signal|csf|white_matter|cosine)')
    all_columns += conf.columns[mask].tolist()
    all_columns = [c for c in all_columns if c in conf.columns] # prevent errors if any of the requested columns were missing
    basicconf = conf[all_columns].copy()
    # add the outliers in single columns
    outlier_cols = conf.filter(like='non_steady_state_outlier')
    non_steady_state_outlier = outlier_cols.any(axis=1).values
    basicconf['non_steady_state_outlier'] = non_steady_state_outlier
    basicconf.to_csv(out_fn,sep='\t',  na_rep='n/a',index=False)

def confound_workflow(fmriprep_out, confounds_dir, filter_=None,fd_threshold=None,std_dvars_threshold=None,min_contiguous=None):
    in_fns = sorted(glob(os.path.join(fmriprep_out, 'func', '*.tsv'))) + \
    sorted(glob(os.path.join(fmriprep_out, 'ses-*', 'func', '*.tsv')))
    if filter_ is not None:
        in_fns = filter_(in_fns)

    for in_fn in in_fns:
        assert '_desc-confounds_' in in_fn
        conf = pd.read_csv(in_fn,delimiter='\t', na_values='n/a')
        conf['framewise_displacement'] = pd.to_numeric(
            conf['framewise_displacement'], errors='coerce'
        )
        # To-do: add filter options for FD and overwrite FD here
        save_basic_confounds(in_fn,conf,confounds_dir=confounds_dir)

        compute_temporal_mask(in_fn,conf,confounds_dir=confounds_dir,fd_threshold=fd_threshold,std_dvars_threshold=std_dvars_threshold,min_contiguous=min_contiguous)
        
        compute_regressors(in_fn,conf,confounds_dir=confounds_dir)

        
        
