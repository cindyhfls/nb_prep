import os
import json
from functools import partial
import numpy as np
import nibabel as nib
import nitransforms as nt
from scipy import ndimage as ndi
from scipy.sparse import hstack as sparse_hstack

from sdcflows.utils.tools import ensure_positive_cosines
from sdcflows.transform import grid_bspline_weights
from sdcflows.utils.epimanip import get_trt

def aligned(aff1: np.ndarray, aff2: np.ndarray) -> bool:
    """Determine if two affines have aligned grids"""
    return np.allclose(
        np.linalg.norm(np.cross(aff1[:-1, :-1].T, aff2[:-1, :-1].T), axis=1),
        0,
        atol=1e-3,
    )

def correct_coordinates(data_shape, coordinates):
    """Handles marginal coordinates for map_coordinates.

    Each voxel occupies a cube, for each dimension, from center - 0.5 to
    center + 0.5 in voxel coordinates. For some coordinates that are close to
    the boundaries, e.g., -0.49 for each dimension, map_coordinates will mark
    them as out-of-bound, though it is still inside the voxel. This function
    will try to handle these cases by setting these marginal coordinates to
    the closest in-bound value for map_coordinates.

    This function is an in-place operation.

    """
    for i, n in enumerate(data_shape[:3]):
        mask = np.logical_and(coordinates[i] >= -0.5, coordinates[i] < 0)
        coordinates[i, mask] = 0
        mask = np.logical_and(coordinates[i] > n - 1, coordinates[i] <= n - 0.5)
        coordinates[i, mask] = n - 1
    return coordinates


def reconstruct_fieldmap_ndarray(
    coefficients: list[nib.Nifti1Image],
    fmap_reference: nib.Nifti1Image,
    target: np.ndarray,
    # transforms: nt.TransformChain,
) -> np.ndarray:
    if not aligned(fmap_reference.affine, coefficients[-1].affine):
        raise ValueError('Reference passed is not aligned with spline grids')
    reference, _ = ensure_positive_cosines(fmap_reference)

    # Generate tensor-product B-Spline weights
    colmat = sparse_hstack(
        [grid_bspline_weights(reference, level) for level in coefficients]
    ).tocsr()
    coefficients = np.hstack(
        [level.get_fdata(dtype='float32').reshape(-1) for level in coefficients]
    )

    # Reconstruct the fieldmap (in Hz) from coefficients
    fmap_ref = np.reshape(colmat @ coefficients, reference.shape[:3])

    # target = transforms.map(target)
    target_ijk = nt.Affine(np.linalg.inv(reference.affine)).map(target)

    fmap_hz = ndi.map_coordinates(fmap_ref, target_ijk.T, order=1, mode='nearest')

    return fmap_hz


def get_source_fn(boldref_json):
    with open(boldref_json) as f:
        orig_fns = json.load(f)['RawSources']
    assert len(orig_fns) == 1
    return orig_fns[0]


def prepare_source(source_fn, layout=None):
    source = nib.load(source_fn)

    if layout is not None:
        meta = layout.files[source_fn].entities
    else:
        meta_fn = source_fn.replace('.nii.gz', '.json')
        if not os.path.exists(meta_fn):
            task = [_ for _ in os.path.basename(meta_fn).split('_') if 'task' in _]
            task = task[0].split('-')[1]
            meta_fn = meta_fn.split('/sub-')[0] + f'/task-{task}_bold.json'
        with open(meta_fn, 'r') as f:
            meta = json.load(f)

    ro_time = get_trt(meta, source_fn)
    pe_dir = meta['PhaseEncodingDirection']

    pe_axis = 'ijk'.index(pe_dir[0])
    pe_flip = pe_dir.endswith('-')
    source, axcodes = ensure_positive_cosines(source)
    axis_flip = axcodes[pe_axis] in 'LPI'
    pe_info = (pe_axis, -ro_time if (axis_flip ^ pe_flip) else ro_time)

    return source, pe_info


def resample_single_volume(
        vol, hmc_xfm, coords_in_ref, shifts_in_ref, shift_idx,
        reduce_func=partial(np.mean, axis=0), post_hoc=None,
        hmc=True, sdc=True, **kwargs):
    config = dict(order=1, mode='nearest',
                  cval=np.nan, prefilter=True)
    config.update(kwargs)

    resampled = []

    # for coords, shifts in zip(coords_in_ref, shifts_in_ref):
    for i, coords in enumerate(coords_in_ref):
        if shifts_in_ref is not None:
            shifts = shifts_in_ref[i]
        shape = coords.shape

        if hmc:
            coords = nib.affines.apply_affine(
                hmc_xfm, coords.reshape(shape[0], -1).T
            ).T.reshape(shape)
        else:
            coords = coords.copy()

        if sdc:
            coords[shift_idx] += shifts

        coords = correct_coordinates(vol.shape, coords)

        result = ndi.map_coordinates(vol, coords, **config)

        resampled.append(result)

    if reduce_func is not None:
        resampled = reduce_func(resampled, axis=0)

    if post_hoc is not None:
        resampled = {key: func(resampled)
                     for key, func in post_hoc.items()}

    return resampled


def resample_volumes(source_data, hmc_xfms, coords_in_ref, shifts_in_ref, shift_idx,
                     reduce_func=partial(np.mean, axis=0), post_hoc=None, **kwargs):
    if post_hoc is None:
        resampled = []
    else:
        resampled = {key: [] for key in post_hoc.keys()}

    # source_data = source.get_fdata()

    for i, vol in enumerate(source_data.transpose(3, 0, 1, 2)):
        hmc_xfm = hmc_xfms[i]

        result = resample_single_volume(
            vol, hmc_xfm, coords_in_ref, shifts_in_ref, shift_idx,
            reduce_func, post_hoc, **kwargs)

        if post_hoc is None:
            resampled.append(result)
        else:
            for key, value in result.items():
                resampled[key].append(value)

    if post_hoc is None:
        resampled = np.stack(resampled, axis=0)
    else:
        for key in resampled.keys():
            resampled[key] = np.stack(resampled[key], axis=0)

    return resampled


def resample_average(source_data, hmc_xfms, coords_in_ref, shifts_in_ref, shift_idx,
                     hmc=True, sdc=True):
    resampled = None

    for i, vol in enumerate(source_data.transpose(3, 0, 1, 2)):
        hmc_xfm = hmc_xfms[i]

        result = resample_single_volume(
            vol, hmc_xfm, coords_in_ref, shifts_in_ref, shift_idx,
            hmc=hmc, sdc=sdc, mode='constant', cval=np.nan)

        if resampled is None:
            resampled = np.zeros_like(result)
            counts = np.zeros_like(result)
        mask = np.isfinite(result)
        resampled[mask] += result[mask]
        counts[mask] += 1

    resampled /= counts

    return resampled


def resample_to_t1w_average(t1w, source, hmc_xfms, ref_to_t1w, ref_to_fmap,
                            fmap_coef, fmap_epi, pe_info, configs, cannon=True,
                            n_frames=None):
    vox2ras = source.affine
    ras2vox = np.linalg.inv(vox2ras)
    hmc_xfms = [ras2vox @ xfm.matrix @ vox2ras for xfm in hmc_xfms]

    shape = t1w.shape
    print(shape)
    coords = np.indices(shape).reshape(len(shape), -1).T
    coords = nt.Affine(t1w.affine).map(coords)
    coords = ref_to_t1w.map(coords)
    coords_in_ref = nt.Affine(ras2vox).map(coords)

    coords_in_fmap = (~ref_to_fmap).map(coords_in_ref)

    if not isinstance(fmap_coef, list):
        fmap_coef = [fmap_coef]
    # print(type(fmap_coef), fmap_coef)
    fmap_hz = reconstruct_fieldmap_ndarray(fmap_coef, fmap_epi, coords_in_fmap)
    shifts_in_ref = fmap_hz * pe_info[1]
    shift_idx = pe_info[0]

    source_data = source.get_fdata()
    if n_frames is not None and source_data.shape[-1] > n_frames:
        frames = np.linspace(0, source_data.shape[-1] - 1, n_frames)
        frames = np.round(frames).astype(np.int32)
        source_data = source_data[..., frames]  

    output = {}
    # for hmc, fmap in [(True, True), (True, False), (False, True), (False, False)]:
    for name, (hmc, sdc) in configs.items():
        resampled = resample_average(source_data, hmc_xfms, [coords_in_ref.T], [shifts_in_ref], shift_idx,
                                     hmc=hmc, sdc=sdc)
        resampled = resampled.reshape(shape)
        img = nib.Nifti1Image(resampled, t1w.affine)
        if cannon:
            img = nib.as_closest_canonical(img)
        output[name] = img
    return output


def resample_to_surface(pial, white, ratios, source, hmc_xfms, ref_to_t1w, ref_to_fmap,
                        fmap_coef, fmap_epi, pe_info,
                        post_hoc=None):
    if isinstance(ratios, int):
        ratios = (np.arange(ratios) + 0.5) / ratios

    vox2ras = source.affine
    ras2vox = np.linalg.inv(vox2ras)
    hmc_xfms = [ras2vox @ xfm.matrix @ vox2ras for xfm in hmc_xfms]

    pial_in_ref, white_in_ref  = [
        nt.Affine(ras2vox).map(ref_to_t1w.map(_)) for _ in (pial, white)]
    coords_in_ref = [
        (white_in_ref * ratio + pial_in_ref * (1 - ratio)).T
        for ratio in ratios]

    if fmap_coef is None:
        shifts_in_ref, shift_idx = None, None
        kwargs = {'sdc': False}
    else:
        pial_in_fmap, white_in_fmap = [
            (~ref_to_fmap).map(_) for _ in (pial_in_ref, white_in_ref)]
        shifts_in_ref = []
        if not isinstance(fmap_coef, list):
            fmap_coef = [fmap_coef]
        for ratio in ratios:
            coords_in_fmap = white_in_fmap * ratio + pial_in_fmap * (1 - ratio)
            fmap_hz = reconstruct_fieldmap_ndarray(fmap_coef, fmap_epi, coords_in_fmap)
            shift = fmap_hz * pe_info[1]
            shifts_in_ref.append(shift)
        shift_idx = pe_info[0]
        kwargs = {}

    source_data = source.get_fdata()

    resampled = resample_volumes(
        source_data, hmc_xfms, coords_in_ref, shifts_in_ref, shift_idx,
        post_hoc=post_hoc, **kwargs)

    return resampled


def find_truncation_boundaries(brainmask, margin=2):
    boundaries = np.zeros((3, 2), dtype=int)
    for dim in range(3):
        mask = np.all(brainmask == 0, axis=tuple(_ for _ in range(3) if _ != dim))
        for i in range(brainmask.shape[dim]):
            if mask[i]:
                boundaries[dim, 0] = i
            else:
                break
        for i in range(brainmask.shape[dim])[::-1]:
            if mask[i]:
                boundaries[dim, 1] = i
            else:
                break
    boundaries[:, 0] -= margin
    boundaries[:, 1] += margin
    boundaries[:, 0] = np.maximum(boundaries[:, 0], 0)
    boundaries[:, 1] = np.minimum(boundaries[:, 1] + 1, brainmask.shape)
    return boundaries


def canonical_volume_coords(brainmask, margin=2, return_affine=False):
    canonical = nib.as_closest_canonical(brainmask)
    boundaries = find_truncation_boundaries(np.asarray(canonical.dataobj), margin=margin)
    coords = np.mgrid[boundaries[0, 0]:boundaries[0, 1], boundaries[1, 0]:boundaries[1, 1], boundaries[2, 0]:boundaries[2, 1], 1:2]
    coords = coords.astype(np.float64)[..., 0]
    coords = np.moveaxis(coords, 0, -1) @ canonical.affine.T
    if return_affine:
        affine = canonical.affine.copy()
        shift = affine[:3, :3] @ boundaries[:, 0]
        affine[:3, 3] += shift
        return coords, affine
    return coords


def canonical_shift_image(
        brainmask, source, ref_to_t1w, ref_to_fmap,
        fmap_coef, fmap_epi, pe_info):
    vox2ras = source.affine
    ras2vox = np.linalg.inv(vox2ras)

    coords, new_affine = canonical_volume_coords(brainmask, return_affine=True)
    shape = coords.shape[:3]
    coords = coords[..., :3].reshape(-1, 3)
    coords = ref_to_t1w.map(coords)
    coords_in_ref = nt.Affine(ras2vox).map(coords)

    coords_in_fmap = (~ref_to_fmap).map(coords_in_ref)
    if not isinstance(fmap_coef, list):
        fmap_coef = [fmap_coef]
    fmap_hz = reconstruct_fieldmap_ndarray(fmap_coef, fmap_epi, coords_in_fmap)
    shifts_in_ref = fmap_hz * pe_info[1]
    shift_idx = pe_info[0]
    img = nib.Nifti1Image(shifts_in_ref.reshape(shape), new_affine)
    return img


def resample_to_canonical_average(
        brainmask, source, hmc_xfms, ref_to_t1w, ref_to_fmap,
        fmap_coef, fmap_epi, pe_info, configs, n_frames=None):
    vox2ras = source.affine
    ras2vox = np.linalg.inv(vox2ras)
    hmc_xfms = [ras2vox @ xfm.matrix @ vox2ras for xfm in hmc_xfms]

    coords, new_affine = canonical_volume_coords(brainmask, return_affine=True)
    shape = coords.shape[:3]
    coords = coords[..., :3].reshape(-1, 3)
    # print(coords.shape)
    coords = ref_to_t1w.map(coords)
    coords_in_ref = nt.Affine(ras2vox).map(coords)

    if fmap_coef is None:
        shifts_in_ref = None
        shift_idx = None
    else:
        coords_in_fmap = (~ref_to_fmap).map(coords_in_ref)

        if not isinstance(fmap_coef, list):
            fmap_coef = [fmap_coef]
        fmap_hz = reconstruct_fieldmap_ndarray(fmap_coef, fmap_epi, coords_in_fmap)
        shifts_in_ref = [fmap_hz * pe_info[1]]
        shift_idx = pe_info[0]

    source_data = source.get_fdata()
    if n_frames is not None and source_data.shape[-1] > n_frames:
        frames = np.linspace(0, source_data.shape[-1] - 1, n_frames)
        frames = np.round(frames).astype(np.int32)
        source_data = source_data[..., frames]  

    output = {}
    # for hmc, fmap in [(True, True), (True, False), (False, True), (False, False)]:
    for name, (hmc, sdc) in configs.items():
        resampled = resample_average(source_data, hmc_xfms, [coords_in_ref.T], shifts_in_ref, shift_idx,
                                     hmc=hmc, sdc=sdc)
        resampled = resampled.reshape(shape)
        img = nib.Nifti1Image(resampled, new_affine)
        output[name] = img
    return output
