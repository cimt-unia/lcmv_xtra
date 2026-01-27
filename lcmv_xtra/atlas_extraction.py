# lcmv_xtra/atlas_extraction.py
import mne
import logging
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from nilearn import datasets, image


def _setup_logger(name, output_dir, verbose=False):
    """Setup logger with file and optional console output."""
    logger = logging.getLogger(f'atlas.{name}')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    
    # File handler (always detailed)
    log_file = Path(output_dir) / f'{name}_atlas_extraction.log'
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler (respects verbose flag)
    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    
    return logger


def _get_mni_coordinates(stc, src):
    """Convert source space coordinates to MNI space."""
    vertices = stc.vertices[0]
    src_rr = src[0]['rr'][vertices] * 1000  # m → mm

    try:
        trans = src[0]['mri_ras_t']['trans']
    except KeyError:
        raise ValueError(
            "Source space missing 'mri_ras_t' transform. "
            "Ensure it's a proper volume source space."
        )

    mni_coords = image.coord_transform(
        src_rr[:, 0], src_rr[:, 1], src_rr[:, 2], trans
    )
    return np.array(mni_coords).T


def _coords_to_voxels(coords_mni, atlas_img):
    """Convert MNI coordinates to voxel indices in atlas space."""
    homog = np.column_stack([coords_mni, np.ones(len(coords_mni))])
    vox_coords = (np.linalg.inv(atlas_img.affine) @ homog.T).T[:, :3]
    return np.round(vox_coords).astype(int)


def _filter_valid_voxels(vox_coords, atlas_shape):
    """Filter voxels that fall within atlas bounds."""
    valid_mask = (
        (vox_coords >= 0).all(axis=1) &
        (vox_coords[:, 0] < atlas_shape[0]) &
        (vox_coords[:, 1] < atlas_shape[1]) &
        (vox_coords[:, 2] < atlas_shape[2])
    )
    return valid_mask


def extract_difumo_ts(stc, src, config, subject_output, logger=None):
    """
    Extract weighted time courses from DiFuMo atlas.
    
    Parameters:
        stc: MNE SourceEstimate
        src: MNE SourceSpaces
        config: dict with 'dimension' and 'resolution_mm'
        subject_output: Output directory path
        logger: Logger instance (optional)
    
    Returns:
        time_courses: (n_components, n_times) array
        component_info: List of dicts with component metadata
    """
    log = logger or logging.getLogger(__name__)
    log.info("Starting DiFuMo atlas extraction")
    
    # Fetch atlas
    log.debug(f"Fetching DiFuMo atlas (dimension={config['dimension']}, "
              f"resolution={config['resolution_mm']}mm)")
    atlas = datasets.fetch_atlas_difumo(
        dimension=config['dimension'],
        resolution_mm=config['resolution_mm']
    )
    atlas_img = nib.load(atlas.maps)
    atlas_shape = atlas_img.shape
    n_components = atlas_shape[3]
    log.info(f"Atlas loaded: {n_components} components, shape {atlas_shape[:3]}")

    # Get MNI coordinates
    log.debug("Converting source coordinates to MNI space")
    src_coords_mni = _get_mni_coordinates(stc, src)
    vertices = stc.vertices[0]
    
    # Convert to voxel coordinates
    vox_coords = _coords_to_voxels(src_coords_mni, atlas_img)
    
    # Filter valid voxels
    valid_mask = _filter_valid_voxels(vox_coords, atlas_shape)
    valid_indices = np.where(valid_mask)[0]
    valid_voxels = vox_coords[valid_mask]
    
    log.info(f"Using {len(valid_indices)}/{len(vertices)} sources within atlas bounds")

    # Prepare data (handle complex values)
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    log.debug(f"STC data shape: {stc_data.shape}, dtype: {stc_data.dtype}")

    # Extract time courses
    time_courses = []
    component_info = []
    threshold = 1e-6

    for comp_idx in range(n_components):
        if comp_idx % 100 == 0:
            log.debug(f"Processing component {comp_idx + 1}/{n_components}")

        try:
            comp_map = atlas_img.slicer[..., comp_idx].get_fdata()
            weights, stc_indices = [], []

            for i, (x, y, z) in enumerate(valid_voxels):
                prob = comp_map[x, y, z]
                if prob > threshold:
                    weights.append(prob)
                    stc_indices.append(valid_indices[i])

            if weights:
                weights = np.array(weights)
                weights /= weights.sum()
                tc = np.average(stc_data[stc_indices], axis=0, weights=weights)
                info = {
                    'component': comp_idx,
                    'n_sources': len(stc_indices),
                    'max_weight': float(weights.max()),
                    'mean_weight': float(weights.mean())
                }
            else:
                tc = np.zeros(stc_data.shape[1])
                info = {
                    'component': comp_idx,
                    'n_sources': 0,
                    'max_weight': 0.0,
                    'mean_weight': 0.0
                }

            time_courses.append(tc)
            component_info.append(info)

        except Exception as e:
            log.warning(f"Error processing component {comp_idx}: {e}")
            time_courses.append(np.zeros(stc_data.shape[1]))
            component_info.append({
                'component': comp_idx,
                'n_sources': 0,
                'max_weight': 0.0,
                'mean_weight': 0.0
            })

    # Summary statistics
    valid_comps = sum(1 for info in component_info if info['n_sources'] > 0)
    log.info(f"Extraction complete: {valid_comps}/{n_components} components assigned")

    # Save outputs
    subject_output = Path(subject_output)
    time_courses_array = np.array(time_courses)
    
    np.save(subject_output / 'difumo_time_courses.npy', time_courses_array)
    pd.DataFrame(component_info).to_csv(
        subject_output / 'difumo_component_info.csv', index=False
    )
    
    log.info(f"Saved time courses: {time_courses_array.shape}")
    log.debug(f"Output directory: {subject_output}")

    return time_courses_array, component_info


def extract_gt_ts(stc, src, gt_atlas_dir, subject_output, logger=None):
    """
    Extract time courses from Glasser+Tian (414 ROI) atlas.
    
    Parameters:
        stc: MNE SourceEstimate
        src: MNE SourceSpaces
        gt_atlas_dir: Path to atlas files directory
        subject_output: Output directory path
        logger: Logger instance (optional)
    
    Returns:
        time_courses: (414, n_times) array
        roi_names: List of 414 ROI names
    """
    log = logger or logging.getLogger(__name__)
    log.info("Starting Glasser+Tian atlas extraction")
    
    # Prepare data
    stc_data = np.abs(stc.data) if np.iscomplexobj(stc.data) else stc.data
    n_times = stc_data.shape[1]
    log.debug(f"STC data shape: {stc_data.shape}")

    # Load atlas files
    gt_dir = Path(gt_atlas_dir)
    glasser_file = gt_dir / "glasser_360_MNI152NLin6Asym.nii.gz"
    tian_file = gt_dir / "tian_subcortex_54_MNI152NLin6Asym.nii"
    
    if not glasser_file.exists():
        raise FileNotFoundError(f"Glasser atlas not found: {glasser_file}")
    if not tian_file.exists():
        raise FileNotFoundError(f"Tian atlas not found: {tian_file}")
    
    log.debug(f"Loading Glasser atlas: {glasser_file}")
    glasser_img = nib.load(glasser_file)
    log.debug(f"Loading Tian atlas: {tian_file}")
    tian_img = nib.load(tian_file)
    
    log.info(f"Glasser shape: {glasser_img.shape}")
    log.info(f"Tian shape (original): {tian_img.shape}")

    # Resample Tian to match Glasser space
    log.debug("Resampling Tian atlas to Glasser space")
    tian_resampled = image.resample_to_img(
        tian_img,
        glasser_img,
        interpolation='nearest',
        force_resample=True,
        copy_header=True
    )
    log.info(f"Tian shape (resampled): {tian_resampled.shape}")

    # Combine atlases
    g_data = glasser_img.get_fdata()
    t_data = tian_resampled.get_fdata()
    
    # Shift Tian labels to avoid overlap (Glasser: 1-360, Tian: 361-414)
    t_data = np.where(t_data > 0, t_data + 360, 0)
    gt_data = g_data + t_data
    gt_img = nib.Nifti1Image(gt_data, affine=glasser_img.affine)
    log.debug("Combined atlas created (Glasser: 1-360, Tian: 361-414)")

    # Get MNI coordinates
    log.debug("Converting source coordinates to MNI space")
    src_coords_mni = _get_mni_coordinates(stc, src)
    vertices = stc.vertices[0]

    # Map to voxel indices
    vox_coords = _coords_to_voxels(src_coords_mni, gt_img)

    # Assign labels to each source
    labels = np.zeros(len(src_coords_mni), dtype=int)
    gt_array = gt_img.get_fdata()
    shape = gt_array.shape

    for i, (x, y, z) in enumerate(vox_coords):
        if (0 <= x < shape[0]) and (0 <= y < shape[1]) and (0 <= z < shape[2]):
            labels[i] = int(gt_array[x, y, z])

    # Extract time courses per ROI
    n_rois = 414
    time_courses = np.full((n_rois, n_times), np.nan, dtype=np.float32)

    for roi in range(1, n_rois + 1):
        mask = (labels == roi)
        if np.any(mask):
            time_courses[roi - 1, :] = np.mean(stc_data[mask, :], axis=0)

    # Load ROI names
    roi_file = gt_dir / "roi_labels.csv"
    if not roi_file.exists():
        raise FileNotFoundError(f"ROI labels file not found: {roi_file}")
    
    roi_df = pd.read_csv(roi_file)
    if len(roi_df) != 414:
        raise ValueError(f"Expected 414 ROIs in roi_labels.csv, got {len(roi_df)}")
    
    roi_names = roi_df['roi_name'].tolist()

    # Clean and save
    time_courses_clean = np.nan_to_num(time_courses, nan=0.0)
    subject_output = Path(subject_output)
    
    np.save(subject_output / "gt_time_courses.npy", time_courses)
    np.save(subject_output / "gt_time_courses_clean.npy", time_courses_clean)
    pd.DataFrame({"roi_name": roi_names}).to_csv(
        subject_output / "gt_roi_names.csv", index=False
    )

    # Summary
    n_assigned = np.sum(~np.isnan(time_courses[:, 0]))
    log.info(f"Extraction complete: {n_assigned}/414 ROIs assigned sources")
    log.info(f"Saved time courses: {time_courses_clean.shape}")
    log.debug(f"Output directory: {subject_output}")

    return time_courses_clean, roi_names

def difumo_extraction(subject_output_dir, global_subjects_dir, 
                      difumo_config=None, verbose=False):
    """
    Run DiFuMo time course extraction on existing source estimates.
    
    Parameters:
        subject_output_dir: Path to subject-specific output directory
        global_subjects_dir: Path to global resources (fsaverage)
        difumo_config: dict with 'dimension' and 'resolution_mm'
        verbose: Enable console logging (default: False)
    
    Returns:
        time_courses: (n_components, n_times) array
        component_info: List of component metadata dicts
    """
    if difumo_config is None:
        difumo_config = {'dimension': 512, 'resolution_mm': 2}

    subject_output_dir = Path(subject_output_dir)
    global_subjects_dir = Path(global_subjects_dir)

    # Setup logger
    subject_name = subject_output_dir.name
    log = _setup_logger(f'{subject_name}_difumo', subject_output_dir, verbose)
    
    log.info(f"{'='*60}")
    log.info(f"DiFuMo Atlas Extraction: {subject_name}")
    log.info(f"{'='*60}")

    stc_file = subject_output_dir / "source_estimate_LCMV.h5"
    if not stc_file.exists():
        raise FileNotFoundError(
            f"STC file not found: {stc_file}\n"
            "Expected H5 format from source estimation pipeline."
        )

    # Load data
    log.info(f"Loading source estimate: {stc_file.name}")
    stc = mne.read_source_estimate(stc_file)
    
    src_file = global_subjects_dir / "fsaverage-vol-5mm-src.fif"
    if not src_file.exists():
        raise FileNotFoundError(f"Source space not found: {src_file}")
    
    log.info(f"Loading source space: {src_file.name}")
    src = mne.read_source_spaces(src_file)
    
    log.info(f"Data loaded: {stc.data.shape[0]} sources × {stc.data.shape[1]} timepoints")

    # Run extraction
    time_courses, component_info = extract_difumo_ts(
        stc=stc,
        src=src,
        config=difumo_config,
        subject_output=subject_output_dir,
        logger=log
    )

    log.info("DiFuMo extraction complete")
    log.info(f"{'='*60}\n")
    
    return time_courses, component_info

def gt_extraction(subject_output_dir, global_subjects_dir, 
                      verbose=False):
    """
    Run Glasser+Tian time course extraction using bundled atlas files.
    
    Parameters:
        subject_output_dir: Path to subject-specific output directory
        global_subjects_dir: Path to global resources (fsaverage)
        verbose: Enable console logging (default: False)
    
    Returns:
        time_courses: (414, n_times) array
        roi_names: List of 414 ROI names
    """
    # Always use the bundled GT atlas files
    import lcmv_xtra
    package_dir = Path(lcmv_xtra.__file__).parent
    gt_atlas_dir = package_dir / 'data' / 'gt_atlas'
    
    if not gt_atlas_dir.exists():
        raise FileNotFoundError(
            f"Bundled GT atlas directory not found at: {gt_atlas_dir}\n")
    
    # Verify required files exist
    required_files = [
        'glasser_360_MNI152NLin6Asym.nii.gz',
        'tian_subcortex_54_MNI152NLin6Asym.nii',
        'roi_labels.csv'
    ]
    for fname in required_files:
        if not (gt_atlas_dir / fname).exists():
            raise FileNotFoundError(f"Required GT atlas file missing: {gt_atlas_dir / fname}")

    subject_output_dir = Path(subject_output_dir)
    global_subjects_dir = Path(global_subjects_dir)

    # Setup logger
    subject_name = subject_output_dir.name
    log = _setup_logger(f'{subject_name}_gt', subject_output_dir, verbose)
    
    log.info(f"{'='*60}")
    log.info(f"Glasser+Tian Atlas Extraction: {subject_name}")
    log.info(f"{'='*60}")

    # Validate files
    stc_file = subject_output_dir / "source_estimate_LCMV.h5"
    if not stc_file.exists():
        raise FileNotFoundError(f"Source estimate not found: {stc_file}")
    
    src_file = global_subjects_dir / "fsaverage-vol-5mm-src.fif"
    if not src_file.exists():
        raise FileNotFoundError(f"Source space not found: {src_file}")

    # Load data
    log.info(f"Loading source estimate: {stc_file.name}")
    stc = mne.read_source_estimate(stc_file)
    
    log.info(f"Loading source space: {src_file.name}")
    src = mne.read_source_spaces(src_file)
    
    log.info(f"Data loaded: {stc.data.shape[0]} sources × {stc.data.shape[1]} timepoints")

    # Run extraction
    time_courses, roi_names = extract_gt_ts(
        stc=stc,
        src=src,
        gt_atlas_dir=gt_atlas_dir,
        subject_output=subject_output_dir,
        logger=log
    )

    log.info("Glasser+Tian extraction complete")
    log.info(f"{'='*60}\n")
    
    return time_courses, roi_names
