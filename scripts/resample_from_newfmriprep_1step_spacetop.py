from nb_prep.resample_workflow_fmriprep_v24_blockavg import dataset_workflow
import neuroboros as nb
import os
import sys

# input directory
fmriprep_out_root = '/dartfs/rc/lab/H/HaxbyLab/feilong/fmriprep_out_root/spacetop_24.1.1/'
bids_dir = os.path.realpath(os.path.expanduser(f"~/lab/BIDS/ds005256-openneuro"))
# output directory
nb_dir = '/dartfs/rc/lab/H/HaxbyLab/datasets/spacetop/24.1.1/'

os.makedirs(nb_dir, exist_ok=True)
dset = nb.SpaceTop(
        fp_version="24.1.1"
    )
sids = dset.subject_sets['movie13']
filter_ = lambda fns: [_ for _ in fns if 'alignvideo' in _ ] # filter for the video runs, delete that argument for all runs

idx = int(sys.argv[1])
dataset_workflow(
    [sids[idx]],
    fmriprep_out_root,
    nb_dir,
    bids_dir=bids_dir,
    n_jobs_subjects=1,
    run_filter=filter_,
    spaces=[],mni_mm=2,atlas_names=['aseg'],
    do_canonical=False)

# if you want surface too, set spaces = ['onavg-ico32','onavg-ico64'] etc.