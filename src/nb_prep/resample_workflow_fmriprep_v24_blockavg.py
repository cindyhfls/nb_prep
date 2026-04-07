import gc
import os
import json
from functools import partial
from glob import glob

from .interfaces.surface import resample_single_volume
import nibabel as nib
import nitransforms as nt
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from nilearn.image import resample_img

from .interfaces.surface import (
    find_truncation_boundaries,
    get_source_fn,
    prepare_source,
    resample_to_surface,
    resample_volumes,
    reconstruct_fieldmap_ndarray,
    resample_to_canonical_average
)
from .resample import parse_combined_hdf5, compute_warp
from .volume import aseg_mapping,resample_mni_to_resolution

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

DIR = os.path.dirname(os.path.abspath(__file__))

MNI_AFFINE_1MM = np.array(
    [[1., 0., 0., -96.],
     [0., 1., 0., -132.],
     [0., 0., 1., -78.],
     [0., 0., 0., 1.]])

MNI_SHAPES = {
    1: (193, 229, 193),
    2: (97, 115, 97),
    3: (65, 77, 65),
    4: (49, 58, 49),
}

# ------------------------------------------------------------------ #
# MNI grid utilities
# ------------------------------------------------------------------ #

def get_mni_coords(mm=2):
    """
    Returns (xyz1, affine, shape) for MNI152NLin2009cAsym at given mm resolution.
    Voxel centers are shifted by -(mm-1)*0.5 to align with block averages of 1mm grid.
    """
    assert mm in MNI_SHAPES, f"mm must be one of {list(MNI_SHAPES.keys())}"
    shape = MNI_SHAPES[mm]

    affine = np.eye(4)
    affine[np.arange(3), np.arange(3)] = mm
    affine[:3, 3] = MNI_AFFINE_1MM[:3, 3] - (mm - 1) * 0.5

    ijk1 = np.concatenate([
        np.tile(np.arange(shape[0])[:, np.newaxis, np.newaxis, np.newaxis],
                (1, shape[1], shape[2], 1)),
        np.tile(np.arange(shape[1])[np.newaxis, :, np.newaxis, np.newaxis],
                (shape[0], 1, shape[2], 1)),
        np.tile(np.arange(shape[2])[np.newaxis, np.newaxis, :, np.newaxis],
                (shape[0], shape[1], 1, 1)),
        np.ones((*shape, 1)),
    ], axis=3)

    xyz1 = ijk1 @ affine.T
    return xyz1, affine, shape


def find_mni_h5(anat_dir, sid):
    """Find the T1w->MNI composite warp h5, handling variable BIDS entities."""
    pattern = os.path.join(
        anat_dir,
        f"sub-{sid}_*from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    )
    fns = glob(pattern)
    assert len(fns) == 1, f"Expected 1 MNI h5, found {len(fns)}: {pattern}"
    return fns[0]


def warp_mni_coords_to_t1w(xyz1, mni_hdf5):
    """
    Apply the T1w->MNI composite warp (h5) in reverse to map
    MNI grid coords into T1w RAS space.
    """
    affine_h5, warp, warp_affine = parse_combined_hdf5(mni_hdf5)
    diff = compute_warp(xyz1, warp, warp_affine, kwargs={'order': 1})
    xyz1 = xyz1.copy()
    xyz1[..., :3] += diff
    xyz1 = xyz1 @ affine_h5.T
    return xyz1


# ------------------------------------------------------------------ #
# Atlas / ROI utilities
# ------------------------------------------------------------------ #

def extract_roi_data(data_4d, atlas_names, mm, onlysubcortex=True):
    """
    Extract ROI time series from data_4d (X, Y, Z, T).

    Parameters
    ----------
    data_4d : np.ndarray (X, Y, Z, T)
    atlas_names : list of str, e.g. ['aseg', 'Tian']
    mm : int, resolution used to load the correct atlas file
    cortex : bool, whether to include cerebral cortex labels (idx 3, 42)

    Returns
    -------
    dict of {roi_name: array (T, N_voxels)}
    """
    target_affine = get_mni_coords(mm=mm)[1]
    target_shape = MNI_SHAPES[mm]
    output = {}

    for atlas_name in atlas_names:
        if atlas_name == 'aseg':
            atlas_img = nib.load(os.path.join(
                DIR, 'atlas',
                f'tpl-MNI152NLin2009cAsym_res-{mm:02d}_atlas-aseg_dseg.nii.gz'
            ))
            if not (np.allclose(atlas_img.affine, target_affine) and
                    atlas_img.shape[:3] == target_shape):
                print(f"Resampling aseg atlas to match {mm}mm MNI grid...")
                atlas_img = resample_img(
                    atlas_img,
                    target_affine=target_affine,
                    target_shape=target_shape,
                    interpolation='nearest',
                )
            atlas = np.asanyarray(atlas_img.dataobj)
            for idx, name in aseg_mapping.items():
                if idx in [3, 42, 8, 47, 16] and onlysubcortex:
                    continue
                mask = (atlas == idx)
                output[name] = data_4d[mask].T  # (T, N_voxels)

        elif atlas_name == 'Tian':
            atlas_img = nib.load(os.path.join(
                DIR, 'atlas', 'Tian_Subcortex_S1_3T_2009cAsym.nii.gz'
            ))
            if not (np.allclose(atlas_img.affine, target_affine) and
                    atlas_img.shape[:3] == target_shape):
                print(f"Resampling Tian atlas to match {mm}mm MNI grid...")
                atlas_img = resample_img(
                    atlas_img,
                    target_affine=target_affine,
                    target_shape=target_shape,
                    interpolation='nearest',
                )
            atlas = np.asanyarray(atlas_img.dataobj)
            mask = (atlas != 0)
            output['Tian_Subcortex'] = data_4d[mask].T  # (T, N_voxels)

        else:
            raise ValueError(
                f'Unsupported atlas: {atlas_name}. '
                f'Supported: "aseg", "Tian"'
            )

    return output


# ------------------------------------------------------------------ #
# MNI volume resampling (parallel structure to resample_to_surface)
# ------------------------------------------------------------------ #

def resample_to_mni(
        mni_coords_in_t1w, source, hmc_xfms,
        ref_to_t1w, ref_to_fmap, fmap_coef, fmap_epi, pe_info,
        mm=2):
    """
    Resample BOLD to MNI volume space.
    Interpolates at 1mm then block-averages to target resolution,
    consistent with original resample_mni_to_resolution approach.
    Processes one volume at a time to avoid memory explosion.

    Parameters
    ----------
    mni_coords_in_t1w : np.ndarray (193, 229, 193, 4)
        1mm MNI grid coords already warped to T1w RAS space
    source : NIfTI image, raw BOLD
    hmc_xfms : nitransforms LinearTransformsMapping
    ref_to_t1w : nitransforms Affine (boldref -> T1w)
    ref_to_fmap, fmap_coef, fmap_epi, pe_info : SDC-related, can be None
    mm : int, target resolution (1, 2, 3, or 4)

    Returns
    -------
    data_4d : np.ndarray (X, Y, Z, T) at target resolution
    affine : np.ndarray (4, 4) affine for target resolution
    """
    vox2ras = source.affine
    ras2vox = np.linalg.inv(vox2ras)

    # Convert hmc xfms to voxel space (same as resample_to_surface)
    hmc_xfms_vox = [ras2vox @ xfm.matrix @ vox2ras for xfm in hmc_xfms]

    # Map 1mm MNI coords (in T1w RAS) -> boldref RAS -> boldref voxels
    coords_flat = mni_coords_in_t1w[..., :3].reshape(-1, 3)  # (N, 3)
    coords_in_ref = nt.Affine(ras2vox).map(
        ref_to_t1w.map(coords_flat))  # (N, 3) in boldref voxels

    if fmap_coef is None:
        shifts_in_ref = None
        shift_idx = None
        kwargs = {'sdc': False}
    else:
        coords_in_fmap = (~ref_to_fmap).map(coords_flat)
        if not isinstance(fmap_coef, list):
            fmap_coef = [fmap_coef]
        fmap_hz = reconstruct_fieldmap_ndarray(fmap_coef, fmap_epi, coords_in_fmap)
        shifts_in_ref = [fmap_hz * pe_info[1]]
        shift_idx = pe_info[0]
        kwargs = {}

    source_data = source.get_fdata()
    n_vols = source_data.shape[-1]
    mni_1mm_shape = MNI_SHAPES[1]   # (193, 229, 193)
    target_shape = MNI_SHAPES[mm]   # e.g. (97, 115, 97) for 2mm

    # pre-allocate output at target resolution
    output = np.empty((n_vols, *target_shape), dtype=np.float64)

    for i, vol in enumerate(source_data.transpose(3, 0, 1, 2)):
        # interpolate single volume to 1mm MNI grid
        result = resample_single_volume(
            vol, hmc_xfms_vox[i],
            [coords_in_ref.T],
            shifts_in_ref, shift_idx,
            **kwargs).reshape(mni_1mm_shape)  # (193, 229, 193)

        # block average single volume to target resolution
        # reuses original resample_mni_to_resolution logic
        downsampled, affine = resample_mni_to_resolution(result, mm=mm)
        output[i] = downsampled  # (target_shape)

    # (T, X, Y, Z) -> (X, Y, Z, T)
    return output.transpose(1, 2, 3, 0), affine

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def find_file(pattern):
    fns = glob(pattern)
    assert len(fns) == 1, f"Expected 1 file, found {len(fns)}: {pattern}"
    return fns[0]

def apply_surface_transform(data, xfm):
    return data @ xfm

def find_anat_dir(root, sid):
    anat_dir = os.path.join(root, "anat")
    if not os.path.exists(anat_dir):
        candidates = glob(f"{root}/ses-*/anat")
        if not candidates:
            raise FileNotFoundError(f"No anat dir found for {sid} in {root}")
        anat_dir = candidates[0]
    return anat_dir

def find_sdc_files(root, sid, ses, xfm_fn):
    fmapid = [_ for _ in xfm_fn.split("_") if _.startswith("to-")][0][3:]
    ref_to_fmap = nt.Affine.from_filename(xfm_fn)
    fmap_coef = nib.load(find_file(
        f"{root}/ses-{ses}/fmap/sub-{sid}_ses-{ses}_*"
        f"_fmapid-{fmapid}_desc-coeff_fieldmap.nii.gz"
    ))
    fmap_epi = nib.load(find_file(
        f"{root}/ses-{ses}/fmap/sub-{sid}_ses-{ses}_*"
        f"_fmapid-{fmapid}_desc-epi_fieldmap.nii.gz"
    ))
    return ref_to_fmap, fmap_coef, fmap_epi


# ------------------------------------------------------------------ #
# Main run workflow
# ------------------------------------------------------------------ #

def run_workflow(
    sid,
    lrs,
    hmc_fn,
    fmriprep_out_dir,
    out_dir,
    bids_dir=None,
    xfm_fn=None,
    do_surface=True,
    do_mni=True,
    do_canonical=True,
    spaces=None,
    mni_mm=2,
    atlas_names=None,
    onlysubcortex=True,
):
    """
    Parameters
    ----------
    sid : str
        Subject ID
    lr : str
        Hemisphere 'l' or 'r' (only used for surface resampling)
    hmc_fn : str
        Path to HMC xfm file
    fmriprep_out_dir : str
        Path to fMRIPrep output sub-{sid} directory
    out_dir : str
        Output directory for resampled data
    bids_dir: str (optional)
        Original bids directory before fmriprep processing
    xfm_fn : str or None
        Path to SDC boldref-to-fmap xfm, None if no SDC
    do_surface : bool
    do_mni : bool
    spaces : list of str
        Surface spaces e.g. ['onavg-ico32', 'onavg-ico64']
    mni_mm : int
        MNI resolution in mm (1, 2, 3, or 4)
    atlas_names : list of str or None
        e.g. ['aseg', 'Tian']
    onlysubcortex : bool
        Whether to include cerebral cortex/cerebellum/brainstem labels in aseg extraction
    """
    if spaces is None:
        spaces = []
    if atlas_names is None:
        atlas_names = []

    root = fmriprep_out_dir
    base = hmc_fn.split("_from-orig_")[0]
    source_fn = get_source_fn(base + "_desc-hmc_boldref.json")
    if source_fn.startswith("bids:raw:"):
        assert bids_dir is not None, "need bids directory input"
        source_fn = f'{bids_dir}/{source_fn.split("bids:raw:")[-1]}'
    out_base = os.path.basename(source_fn).rsplit("_bold.nii.gz", 1)[0] + ".npy"

    ses = os.path.relpath(hmc_fn, root).split("/")[0]
    ses = ses[4:] if ses.startswith("ses-") else None

    # ------------------------------------------------------------------ #
    # Locate anat dir
    # ------------------------------------------------------------------ #
    anat_dir = find_anat_dir(root, sid)

    # ------------------------------------------------------------------ #
    # Shared transforms
    # ------------------------------------------------------------------ #
    ref_to_t1w = nt.Affine.from_filename(
        base + "_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"
    )
    hmc_xfms = nt.LinearTransformsMapping.from_filename(
        base + "_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt"
    )
    source, pe_info = prepare_source(source_fn)

    if xfm_fn is not None:
        ref_to_fmap, fmap_coef, fmap_epi = find_sdc_files(root, sid, ses, xfm_fn)
    else:
        ref_to_fmap, fmap_coef, fmap_epi = None, None, None
    # ------------------------------------------------------------------ #
    # Surface resampling
    # ------------------------------------------------------------------ #
    if do_surface:
        for lr in lrs:
            post_hoc = {}  # reset per hemisphere
            for space in spaces:
                out_fn = (
                    f"{out_dir}/resampled/{space}/{lr}-cerebrum/1step_pial_overlap/{out_base}"
                )
                if os.path.exists(out_fn):
                    continue
                xforms_dir = os.path.join(out_dir, "xforms")
                if not os.path.exists(xforms_dir):
                    xforms_dir = os.path.join(out_dir, "xfms")
                if not os.path.exists(xforms_dir):
                    raise FileNotFoundError(f"Xforms dir not found: {xforms_dir}")
                xform = sparse.load_npz(
                    f"{xforms_dir}/{space}/{sid}_overlap_{lr}h.npz")
                xform = sparse.diags(
                    np.reciprocal(xform.sum(axis=1).A.ravel() + 1e-10)
                ) @ xform
                post_hoc[space] = partial(apply_surface_transform, xfm=xform)

            if post_hoc:
                pial, white = (
                    nib.load(glob(
                        f"{anat_dir}/sub-{sid}_*hemi-{lr.upper()}_{kind}.surf.gii")[0])
                    .darrays[0].data.astype(np.float64)
                    for kind in ("pial", "white")
                )
                resampled = resample_to_surface(
                    pial, white, 5, source, hmc_xfms,
                    ref_to_t1w, ref_to_fmap, fmap_coef, fmap_epi, pe_info,
                    post_hoc=post_hoc,
                )
                # resampled is a dict {space: array}
                for space, data in resampled.items():
                    out_fn = (
                        f"{out_dir}/resampled/{space}/{lr}-cerebrum/1step_pial_overlap/{out_base}"
                    )
                    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
                    np.save(out_fn, data)
                    print(out_fn, data.shape, data.dtype)
                del resampled, pial, white
                gc.collect()
    # ------------------------------------------------------------------ #
    # MNI volume resampling (only run once, when lr == 'l')
    # ------------------------------------------------------------------ #
    if do_mni:
        print("MNI volume")
        tag = "1step_linear_overlap_blockavg" #tag = "1step_linear_overlap_blockavg"
        space = f"mni-{mni_mm}mm"

        nifti_out_fn = (
            f"{out_dir}/resampled/{space}/average-volume/{tag}/"
            f"{out_base[:-4]}"
            f"_space-MNI152NLin2009cAsym_res-{mni_mm:02d}.nii.gz"
        )
        roi_out_fns = []
        if atlas_names:
            # pre-compute expected output filenames to check if we can skip
            for atlas_name in atlas_names:
                if atlas_name == 'aseg':
                    for name in aseg_mapping.values():
                        roi_out_fns.append(
                            f"{out_dir}/resampled/{space}/{name}/{tag}/"
                            f"{out_base[:-4]}.npy"
                        )
                elif atlas_name == 'Tian':
                    roi_out_fns.append(
                        f"{out_dir}/resampled/{space}/Tian_Subcortex/{tag}/"
                        f"{out_base[:-4]}.npy"
                    )

        if (os.path.exists(nifti_out_fn) and
                all(os.path.exists(f) for f in roi_out_fns)):
            print(f"All MNI outputs exist for {out_base}, skipping.")
            return
        # always use 1mm coords for interpolation
        xyz1, _, _ = get_mni_coords(mm=1)
        mni_hdf5 = find_mni_h5(anat_dir, sid)
        mni_coords_in_t1w = warp_mni_coords_to_t1w(xyz1, mni_hdf5)

        data_4d, mni_affine_mm = resample_to_mni(
            mni_coords_in_t1w, source, hmc_xfms,
            ref_to_t1w, ref_to_fmap, fmap_coef, fmap_epi, pe_info,
            mm=mni_mm,
        )
        gc.collect()
        # # Prepare MNI coords warped to T1w space
        # xyz1, mni_affine_mm, mni_shape = get_mni_coords(mm=mni_mm)
        # mni_hdf5 = find_mni_h5(anat_dir, sid)
        # mni_coords_in_t1w = warp_mni_coords_to_t1w(xyz1, mni_hdf5)

        # # Resample BOLD to MNI # the original resample_volumes is too much memory
        # data_4d = resample_to_mni(
        #     mni_coords_in_t1w, mni_shape, source, hmc_xfms,
        #     ref_to_t1w, ref_to_fmap, fmap_coef, fmap_epi, pe_info,
        # )
        # data_4d shape: (X, Y, Z, T)

        # Save full NIfTI
        os.makedirs(os.path.dirname(nifti_out_fn), exist_ok=True)
        img = nib.Nifti1Image(np.mean(data_4d, axis=3), mni_affine_mm)
        img.to_filename(nifti_out_fn)
        print(nifti_out_fn, img.shape, data_4d.dtype)

        # ROI extraction
        if atlas_names:
            roi_data = extract_roi_data(data_4d, atlas_names, mni_mm, onlysubcortex=onlysubcortex)
            for roi_name, ts in roi_data.items():
                out_fn = (
                    f"{out_dir}/resampled/{space}/{roi_name}/{tag}/"
                    f"{out_base[:-4]}.npy"
                )
                os.makedirs(os.path.dirname(out_fn), exist_ok=True)
                np.save(out_fn, ts)
                print(out_fn, ts.shape, ts.dtype)
                del ts
                gc.collect()
        del data_4d
        gc.collect()
    if do_canonical:
        mask_fn = glob(f"{anat_dir}/sub-{sid}*_desc-brain_mask.nii.gz")[0]
        brainmask = nib.as_closest_canonical(nib.load(mask_fn))

        # single fully-corrected output
        configs = {'hmc_sdc' if xfm_fn is not None else 'hmc_nosdc': (True, xfm_fn is not None)}

        out_fn = (
            f"{out_dir}/resampled/canonical/average-volume/1step_linear_overlap/"
            f"{out_base[:-4]}.nii.gz"
        )
        if not os.path.exists(out_fn):
            os.makedirs(os.path.dirname(out_fn), exist_ok=True)
            output = resample_to_canonical_average(
                brainmask, source, hmc_xfms,
                ref_to_t1w, ref_to_fmap,
                fmap_coef, fmap_epi, pe_info,
                configs,
            )
            key = list(output.keys())[0]
            nib.save(output[key], out_fn)
            print(out_fn)
            del output
            gc.collect()
# ------------------------------------------------------------------ #
# Subject-level workflow
# ------------------------------------------------------------------ #

def subject_workflow(
    sid,
    fmriprep_out_dir,
    out_dir,
    run_filter=None,
    **kwargs,
):
    """
    Parameters
    ----------
    sid : str
    fmriprep_out_dir : str
        Path to fMRIPrep output sub-{sid} directory
    out_dir : str
    run_filter : callable or None
        Optional function to filter hmc_fns list

    **kwargs : passed to run_workflow
    """
    hmc_fns = sorted(glob(
        f"{fmriprep_out_dir}/ses-*/func/sub-*"
        f"_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt"
    )) + sorted(glob(
        f"{fmriprep_out_dir}/func/sub-*"
        f"_from-orig_to-boldref_mode-image_desc-hmc_xfm.txt"
    ))

    if run_filter is not None:
        hmc_fns = run_filter(hmc_fns)

    print(f"{sid}: {len(hmc_fns)} runs found")

    do_surface = kwargs.get('do_surface', True)
    lrs = 'lr' if do_surface else ''

    for hmc_fn in hmc_fns:
        try:
            run_workflow(
                sid, lrs, hmc_fn,
                fmriprep_out_dir=fmriprep_out_dir,
                out_dir=out_dir,
                **kwargs,
            )
        except FileNotFoundError as e:
            print(e)
            continue


# ------------------------------------------------------------------ #
# Dataset-level workflow
# ------------------------------------------------------------------ #

def dataset_workflow(
    sids,
    fmriprep_out_root,
    out_dir,
    n_jobs_subjects=-1,
    **kwargs,
):
    """
    Parameters
    ----------
    sids : list of str
    fmriprep_out_root : str
        Root dir containing per-subject fMRIPrep output folders
    out_dir : str
    n_jobs_subjects : int
        Parallelism across subjects (-1 = all cores)

    **kwargs : passed to subject_workflow
    """
    def _run(sid):
        fmriprep_out_dir = os.path.join(fmriprep_out_root, f"sub-{sid}")
        if not os.path.exists(fmriprep_out_dir):
            fmriprep_out_dir = os.path.join(fmriprep_out_root, f"output_{sid}/sub-{sid}")
        assert os.path.exists(fmriprep_out_dir), f"fMRIPrep output dir not found for {sid}: {fmriprep_out_dir}"
        subject_workflow(
            sid,
            fmriprep_out_dir=fmriprep_out_dir,
            out_dir=out_dir,
            **kwargs,
        )

    jobs = [delayed(_run)(sid) for sid in sids]
    with Parallel(n_jobs=n_jobs_subjects, verbose=1) as parallel:
        parallel(jobs)