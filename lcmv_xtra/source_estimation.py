# lcmv_xtra/source_estimation.py
import mne
import logging
import numpy as np
from pathlib import Path
from .utils import parse_gpsc

# MNE logging
mne.set_log_level('warning')

# BEL 280-channel system constants
_BEL_CHANNEL_MAP = {str(i): f'E{i}' for i in range(1, 281)}
_BEL_CHANNEL_MAP['REF CZ'] = 'Cz'
_REQUIRED_FIDUCIALS = ['FidNz', 'FidT9', 'FidT10']


def _setup_logger(subject_id, task, output_dir, verbose=False):
    """Setup per-subject logger with file and optional console output."""
    logger = logging.getLogger(f'lcmv.{subject_id}.{task}')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    
    # File handler (always detailed)
    log_file = Path(output_dir) / f'{subject_id}_{task}_processing.log'
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


def load_subject(ica_file_path, gpsc_file_path, subject_id=None, logger=None):
    """
    Load and preprocess subject data for source estimation.
    
    Parameters:
        ica_file_path: Path to ICA-cleaned FIF file
        gpsc_file_path: Path to .gpsc file  
        subject_id: Optional subject ID for logging
        logger: Logger instance (optional)
    
    Returns:
        raw: Preprocessed MNE Raw object
        ch_pos: Dictionary of channel positions
    """
    log = logger or logging.getLogger(__name__)
    ica_file = Path(ica_file_path)
    gpsc_file = Path(gpsc_file_path)
    
    # Validate files
    if not ica_file.exists():
        raise FileNotFoundError(f"ICA file not found: {ica_file}")
    if not gpsc_file.exists():
        raise FileNotFoundError(f"GPSC file not found: {gpsc_file}")

    # Load raw data
    raw = mne.io.read_raw_fif(ica_file, preload=True)
    sfreq = raw.info['sfreq']
    duration_min = raw.n_times / sfreq / 60
    subject_str = f" ({subject_id})" if subject_id else ""
    log.info(f"Loaded data: {duration_min:.1f}min @ {sfreq}Hz{subject_str}")

    # Rename channels
    existing_channels = set(raw.info['ch_names'])
    valid_channel_map = {
        old: new for old, new in _BEL_CHANNEL_MAP.items()
        if old in existing_channels
    }
    
    if valid_channel_map:
        raw.rename_channels(valid_channel_map)
        log.debug(f"Renamed {len(valid_channel_map)} channels")

    # Parse and normalize coordinates
    channels = parse_gpsc(gpsc_file)
    if not channels:
        raise ValueError("No valid channels found in .gpsc file")
    
    gpsc_array = np.array([ch[1:4] for ch in channels])
    mean_pos = np.mean(gpsc_array, axis=0)
    log.debug(f"Original mean position (mm): {mean_pos}")
    
    # Normalize to center origin and convert to meters
    channels_normalized = [
        (ch[0], ch[1] - mean_pos[0], ch[2] - mean_pos[1], ch[3] - mean_pos[2]) 
        for ch in channels
    ]
    ch_pos = {ch[0]: np.array(ch[1:4]) / 1000.0 for ch in channels_normalized}
    
    # Validate fiducials
    missing_fids = [fid for fid in _REQUIRED_FIDUCIALS if fid not in ch_pos]
    if missing_fids:
        raise ValueError(f"Missing required fiducials: {missing_fids}")

    # Create and apply montage
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=ch_pos['FidNz'],
        lpa=ch_pos['FidT9'],
        rpa=ch_pos['FidT10'],
        coord_frame='head'
    )
    raw.set_montage(montage, on_missing='warn')
    raw = raw.pick(['eeg', 'stim'], exclude=raw.info['bads'])
    log.info(f"Applied montage with {len(ch_pos)} positions")

    # Handle average reference
    has_avg_ref = any(p['desc'] == 'average' for p in raw.info['projs'])
    if not has_avg_ref:
        log.debug("Applying average reference projection")
        raw.set_eeg_reference('average', projection=True)
    
    if not raw.proj:
        raw.apply_proj()
        log.debug("Applied EEG projections")

    log.info("Preprocessing complete")
    return raw, ch_pos


def _run_coregistration(raw, ch_pos, subject, subjects_dir, trans_file, logger):
    """Run enhanced coregistration with ICP and outlier removal."""
    log = logger or logging.getLogger(__name__)
    
    coreg = mne.coreg.Coregistration(
        raw.info,
        subject=subject,
        subjects_dir=subjects_dir,
        fiducials={
            'nasion': ch_pos['FidNz'],
            'lpa': ch_pos['FidT9'],
            'rpa': ch_pos['FidT10']
        }
    )

    # Fit fiducials
    log.debug("Fitting fiducials...")
    coreg.fit_fiducials(verbose=False)

    # Initial ICP
    log.debug("Running initial ICP with EEG channels...")
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=False)
    
    # Remove outliers
    dists = coreg.compute_dig_mri_distances()
    n_excluded = np.sum(dists > 5.0/1000)
    if n_excluded > 0:
        log.debug(f"Excluding {n_excluded} outlier points (>5mm)")
        coreg.omit_head_shape_points(distance=5.0/1000)
        
    # Final refinement
    log.debug("Final ICP refinement...")
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=False)

    # Save and compute errors
    trans = coreg.trans
    mne.write_trans(trans_file, trans, overwrite=True)
    
    dists = coreg.compute_dig_mri_distances() * 1000  # mm
    mean_err, median_err, max_err = np.mean(dists), np.median(dists), np.max(dists)
    log.info(f"Coregistration error (mm) - Mean: {mean_err:.2f}, Median: {median_err:.2f}, Max: {max_err:.2f}")
    
    if mean_err > 5.0:
        log.warning(f"Mean coregistration error {mean_err:.2f}mm exceeds 5mm threshold")
    
    return trans, {'mean': mean_err, 'median': median_err, 'max': max_err}


def lcmv_beamformer(raw, ch_pos, project_base, subject_id, task, 
                    reg=0.01, n_jobs=1, verbose=False):
    """
    Run LCMV source estimation with minimal output (clean mode).
    Saves only essential files for downstream analysis.
    """
    project_base = Path(project_base)
    global_subjects_dir = project_base / 'derivatives/lcmv'
    subject_output = project_base / f'derivatives/lcmv/{subject_id}_{task}'
    subject_output.mkdir(parents=True, exist_ok=True)
    
    log = _setup_logger(subject_id, task, subject_output, verbose)
    log.info(f"{'='*60}")
    log.info(f"LCMV Source Estimation (Clean Mode): {subject_id} - {task}")
    log.info(f"{'='*60}")

    # Validate required global files
    subject = 'fsaverage'
    bem_file = global_subjects_dir / 'fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif'
    src_file = global_subjects_dir / 'fsaverage-vol-5mm-src.fif'

    for f in [bem_file, src_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required fsaverage file missing: {f}")

    # Coregistration
    log.info("Running coregistration...")
    trans_file = subject_output / 'fsaverage-trans.fif'
    trans, coreg_errors = _run_coregistration(
        raw, ch_pos, subject, global_subjects_dir, trans_file, log
    )

    # Source space (global, already exists)
    log.info("Loading source space...")
    src = mne.read_source_spaces(src_file)
    n_active = len(src[0]['vertno'])
    log.info(f"Source space: {n_active} active sources")

    # Forward solution
    log.info("Computing forward solution...")
    fwd_file = subject_output / 'fsaverage-vol-eeg-fwd.fif'
    bem = mne.read_bem_solution(bem_file)
    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=src, bem=bem, 
        eeg=True, mindist=5.0, n_jobs=n_jobs
    )
    mne.write_forward_solution(fwd_file, fwd, overwrite=True)

    # LCMV beamformer
    log.info("Computing covariance and LCMV filters...")
    cov = mne.compute_raw_covariance(
        raw, method='oas', picks='eeg', rank='info', n_jobs=n_jobs, verbose=False
    )

    filters = mne.beamformer.make_lcmv(
        info=raw.info, forward=fwd, data_cov=cov, noise_cov=cov, reg=reg,
        pick_ori='max-power', weight_norm='unit-noise-gain', 
        reduce_rank=True, rank='info', verbose=False
    )

    log.info("Applying LCMV beamformer to raw data...")
    stc = mne.beamformer.apply_lcmv_raw(raw=raw, filters=filters)
    stc_file = subject_output / 'source_estimate_LCMV.h5'
    stc.save(stc_file, ftype='h5', overwrite=True)
    log.info(f"Saved source estimate: {stc.data.shape[0]} × {stc.data.shape[1]}")

    # Minimal metadata as JSON (optional but recommended)
    import json
    metadata = {
        'subject_id': subject_id,
        'task': task,
        'sfreq_hz': float(raw.info['sfreq']),
        'duration_min': float(raw.n_times / raw.info['sfreq'] / 60),
        'n_sources': int(stc.data.shape[0]),
        'n_timepoints': int(stc.data.shape[1]),
        'coreg_mean_error_mm': float(coreg_errors['mean']),
        'regularization': reg
    }
    with open(subject_output / 'pipeline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Clean-mode source estimation complete: {subject_output}")
    log.info(f"{'='*60}\n")
    
    return metadata


def execute_source_estimation(project_base, subject_id, task, 
                              ica_file_path,
                              reg=0.01, n_jobs=1, verbose=False):
    import lcmv_xtra
    package_dir = Path(lcmv_xtra.__file__).parent
    gpsc_full_path = package_dir / 'data' / 'bel_280' / 'ghw280_from_egig.gpsc'
    
    if not gpsc_full_path.exists():
        raise FileNotFoundError(f"Bundled .gpsc file not found at: {gpsc_full_path}")
    
    ica_full_path = Path(project_base) / ica_file_path

    # No need to mkdir here — lcmv_beamformer handles it
    log = _setup_logger(subject_id, task, Path(project_base) / f'derivatives/lcmv/{subject_id}_{task}', verbose)
    
    log.info(f"Using bundled BEL 280 .gpsc file: {gpsc_full_path.name}")
    log.info("Loading subject data...")
    raw, ch_pos = load_subject(
        ica_file_path=ica_full_path,
        gpsc_file_path=gpsc_full_path,
        subject_id=subject_id,
        logger=log
    )
    
    return lcmv_beamformer(
        raw=raw,
        ch_pos=ch_pos,
        project_base=project_base,
        subject_id=subject_id,
        task=task,
        reg=reg,
        n_jobs=n_jobs,
        verbose=verbose
    )
