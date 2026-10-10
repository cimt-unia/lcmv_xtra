# lcmv_xtra/source_estimation_epochs.py
import mne
import json
import logging
import lcmv_xtra
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from .utils import parse_gpsc

# MNE logging
mne.set_log_level('warning')

# BEL 280-channel system constants
_BEL_CHANNEL_MAP = {str(i): f'E{i}' for i in range(1, 281)}
_BEL_CHANNEL_MAP['REF CZ'] = 'Cz'
_REQUIRED_FIDUCIALS = ['FidNz', 'FidT9', 'FidT10']


def _setup_logger(subject_id: str, task: str, output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Setup per-subject logger with file and optional console output."""
    logger = logging.getLogger(f'lcmv.{subject_id}.{task}')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()

    log_file = output_dir / f'{subject_id}_{task}_processing.log'
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)

    return logger


def load_epochs(epochs_file_path: Path, gpsc_file_path: Path,
                subject_id: Optional[str] = None,
                logger: Optional[logging.Logger] = None) -> Tuple[mne.Epochs, Dict]:
    """Load epoched data and ALWAYS apply fresh montage from GPSC.

    The montage is unconditionally re-applied from the GPSC file to ensure
    consistency with coregistration. Interpolated channels in stored epochs
    may carry stale digitization positions that degrade ICP alignment.
    This matches the behavior of the original load_subject().
    """
    log = logger or logging.getLogger(__name__)
    epochs_file = Path(epochs_file_path)
    gpsc_file = Path(gpsc_file_path)

    if not epochs_file.exists():
        raise FileNotFoundError(f"Epochs file not found: {epochs_file}")
    if not gpsc_file.exists():
        raise FileNotFoundError(f"GPSC file not found: {gpsc_file}")

    epochs = mne.read_epochs(epochs_file, preload=True, verbose='WARNING')
    sfreq = epochs.info['sfreq']
    subject_str = f" ({subject_id})" if subject_id else ""
    log.info(f"Loaded epochs: {len(epochs)} trials, {epochs.tmin:.2f} to {epochs.tmax:.2f}s @ {sfreq}Hz{subject_str}")

    # Parse GPSC for fiducials and channel positions
    channels = parse_gpsc(gpsc_file)
    if not channels:
        raise ValueError("No valid channels found in .gpsc file")

    gpsc_array = np.array([ch[1:4] for ch in channels])
    mean_pos = np.mean(gpsc_array, axis=0)
    ch_pos = {
        ch[0]: np.array([ch[1] - mean_pos[0], ch[2] - mean_pos[1], ch[3] - mean_pos[2]]) / 1000.0
        for ch in channels
    }

    missing_fids = [fid for fid in _REQUIRED_FIDUCIALS if fid not in ch_pos]
    if missing_fids:
        raise ValueError(f"Missing required fiducials in GPSC: {missing_fids}")

    # ALWAYS re-apply montage from GPSC — never trust stored montage in epochs.
    # This matches original load_subject() behavior and ensures coregistration
    # uses consistent sensor positions regardless of prior interpolation.
    log.info("Applying fresh montage from GPSC (ensures coregistration consistency)...")
    existing_channels = set(epochs.info['ch_names'])
    valid_channel_map = {old: new for old, new in _BEL_CHANNEL_MAP.items() if old in existing_channels}
    if valid_channel_map:
        epochs.rename_channels(valid_channel_map)

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos, nasion=ch_pos['FidNz'], lpa=ch_pos['FidT9'], rpa=ch_pos['FidT10'], coord_frame='head'
    )
    epochs.set_montage(montage, on_missing='warn')

    # Ensure average reference is applied if not already
    has_avg_ref = any(p['desc'] == 'average' for p in epochs.info['projs'])
    if not has_avg_ref and 'eeg' in epochs:
        epochs.set_eeg_reference('average', projection=True)
    if not epochs.proj and 'eeg' in epochs:
        epochs.apply_proj()

    log.info("Epochs loading and validation complete")
    return epochs, ch_pos


def validate_fsaverage(subjects_dir: Path) -> Tuple[Path, Path]:
    """Validate fsaverage resources."""
    subjects_dir = Path(subjects_dir)
    fsaverage_dir = subjects_dir / 'fsaverage'
    bem_file = fsaverage_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
    src_file = subjects_dir / 'fsaverage-vol-5mm-src.fif'

    if not (bem_file.exists() and src_file.exists()):
        raise FileNotFoundError(f"Missing fsaverage files in {subjects_dir}")
    return bem_file, src_file


def _run_coregistration(info: mne.Info, ch_pos: Dict, subject: str,
                        subjects_dir: Path, trans_file: Path,
                        logger: logging.Logger) -> Tuple[mne.transforms.Transform, Dict]:
    """Run enhanced coregistration with ICP and outlier removal."""
    log = logger or logging.getLogger(__name__)
    coreg = mne.coreg.Coregistration(
        info, subject=subject, subjects_dir=subjects_dir,
        fiducials={'nasion': ch_pos['FidNz'], 'lpa': ch_pos['FidT9'], 'rpa': ch_pos['FidT10']}
    )
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=False)

    dists = coreg.compute_dig_mri_distances()
    if np.sum(dists > 5.0 / 1000) > 0:
        coreg.omit_head_shape_points(distance=5.0 / 1000)

    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=False)
    trans = coreg.trans
    mne.write_trans(trans_file, trans, overwrite=True)

    dists = coreg.compute_dig_mri_distances() * 1000
    mean_err, median_err, max_err = np.mean(dists), np.median(dists), np.max(dists)
    log.info(f"Coreg error (mm) - Mean: {mean_err:.2f}, Median: {median_err:.2f}, Max: {max_err:.2f}")

    if mean_err > 5.0:
        raise RuntimeError(f"Mean error {mean_err:.2f}mm exceeds 5mm threshold.")

    return trans, {'mean': mean_err, 'median': median_err, 'max': max_err}


def lcmv_beamformer_epochs(
    epochs: mne.Epochs,
    ch_pos: Dict,
    fsaverage_dir: Path,
    output_dir: Path,
    subject_id: str,
    task: str,
    reg: float = 0.05,
    n_jobs: int = 1,
    verbose: bool = False,
    noise_cov_method: str = 'shrunk',
    baseline_tmin: Optional[float] = None,
    baseline_tmax: Optional[float] = 0.0,
    data_cov_tmin: Optional[float] = None,
    data_cov_tmax: Optional[float] = None,
    compute_noise_cov: bool = True,
) -> Dict:
    """
    Run epoch-based LCMV source estimation for single-trial time series.

    Accepts pre-defined Epochs, computes data covariance (and optionally noise
    covariance) from manually specified time windows, and applies whitened LCMV
    filters via apply_lcmv_epochs.

    NOTE: For single-trial time series extraction, data_cov and noise_cov are
    passed SEPARATELY to make_lcmv. The common_cov addition trick (data + noise)
    is ONLY valid for source power maps (apply_lcmv_cov), not for time series.
    Using common_cov with noise_cov for time series causes double-whitening that
    destroys oscillatory signals like beta ERD.
    """
    fsaverage_dir = Path(fsaverage_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logger(subject_id, task, output_dir, verbose)
    log.info(f"{'='*60}")
    log.info(f"LCMV Epoch Source Estimation: {subject_id} - {task}")
    log.info(f"{'='*60}")

    # 1. Coregistration & Forward
    bem_file, src_file = validate_fsaverage(fsaverage_dir)
    trans_file = output_dir / 'fsaverage-trans.fif'
    trans, coreg_errors = _run_coregistration(epochs.info, ch_pos, 'fsaverage', fsaverage_dir, trans_file, log)

    src = mne.read_source_spaces(src_file)
    fwd_file = output_dir / 'fsaverage-vol-eeg-fwd.fif'
    bem = mne.read_bem_solution(bem_file)
    fwd = mne.make_forward_solution(
        epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=n_jobs
    )
    mne.write_forward_solution(fwd_file, fwd, overwrite=True)

    log.info(f"Using provided epochs: {len(epochs)} trials, {epochs.tmin:.2f}s to {epochs.tmax:.2f}s")
    if len(epochs) == 0:
        raise RuntimeError("No epochs provided or all epochs were dropped.")

    epochs_eeg = epochs.copy().pick('eeg')
    log.info(f"Final epoch count: {len(epochs)}")

    # 2. Compute DATA Covariance (Manual Window)
    log.info(f"Computing DATA covariance from window [tmin={data_cov_tmin}, tmax={data_cov_tmax}]s using method='oas'...")
    data_cov = mne.compute_covariance(
        epochs_eeg, tmin=data_cov_tmin, tmax=data_cov_tmax,
        method='oas', rank=None, n_jobs=n_jobs, verbose=False
    )
    data_rank = mne.compute_rank(data_cov, info=epochs_eeg.info)
    log.info(f"Data covariance rank: {data_rank}")

    # 3. Compute NOISE Covariance (Optional, Manual Window)
    noise_cov = None
    noise_rank = None
    if compute_noise_cov:
        log.info(f"Computing NOISE covariance from baseline [tmin={baseline_tmin}, tmax={baseline_tmax}]s using method='{noise_cov_method}'...")
        noise_cov = mne.compute_covariance(
            epochs_eeg, tmin=baseline_tmin, tmax=baseline_tmax,
            method=noise_cov_method, rank=None, n_jobs=n_jobs, verbose=False
        )
        noise_rank = mne.compute_rank(noise_cov, info=epochs_eeg.info)
        log.info(f"Noise covariance rank: {noise_rank}")
        log.info("Noise covariance will be used for spatial whitening (separate from data_cov).")
    else:
        log.info("Skipping noise covariance computation. Using data covariance only.")

    # 4. Make LCMV filters
    # CRITICAL: For single-trial time series (apply_lcmv_epochs), pass data_cov
    # and noise_cov SEPARATELY. Do NOT add them together.
    # The common_cov = data_cov + noise_cov trick is ONLY for source power maps
    # (apply_lcmv_cov) where active/baseline power ratios cancel depth bias.
    # For time series, adding them causes double-whitening that destroys ERD/ERS.
    log.info("Computing LCMV filters for single-trial time series extraction...")
    filters = mne.beamformer.make_lcmv(
        info=epochs.info, forward=fwd,
        data_cov=data_cov,      # Defines the signal topography to pass through
        noise_cov=noise_cov,    # Defines the spatial whitening (baseline noise profile)
        reg=reg,
        pick_ori='max-power',
        weight_norm='unit-noise-gain',  # Mandatory for time-series to fix depth bias
        reduce_rank=True,
        rank=None,
        verbose=False
    )

    # 5. Apply LCMV to epochs (Single-Trial Time Series)
    log.info("Applying LCMV beamformer to single-trial epochs...")
    stcs: List[mne.VolSourceEstimate] = mne.beamformer.apply_lcmv_epochs(
        epochs=epochs, filters=filters
    )

    # 6. Save each epoch STC individually
    log.info(f"Saving {len(stcs)} epoch source estimates...")
    for i, stc in enumerate(stcs):
        stc_file = output_dir / f'source_estimate_LCMV_epoch_{i:03d}.h5'
        stc.save(stc_file, ftype='h5', overwrite=True)

    # Metadata
    metadata = {
        'subject_id': subject_id,
        'task': task,
        'sfreq_hz': float(epochs.info['sfreq']),
        'epoch_tmin': float(epochs.tmin),
        'epoch_tmax': float(epochs.tmax),
        'n_epochs': len(stcs),
        'n_sources': int(stcs[0].data.shape[0]),
        'n_timepoints_per_epoch': int(stcs[0].data.shape[1]),
        'coreg_mean_error_mm': float(coreg_errors['mean']),
        'regularization': reg,
        'data_covariance_method': 'epoch_averaged_oas',
        'data_covariance_window': [float(data_cov_tmin) if data_cov_tmin is not None else None,
                                   float(data_cov_tmax) if data_cov_tmax is not None else None],
        'compute_noise_covariance': compute_noise_cov,
        'noise_covariance_method': noise_cov_method if compute_noise_cov else None,
        'noise_baseline_window': [float(baseline_tmin) if baseline_tmin is not None else None,
                                  float(baseline_tmax) if baseline_tmax is not None else None] if compute_noise_cov else None,
        'data_rank': data_rank,
        'noise_rank': noise_rank,
        'weight_normalization': 'unit-noise-gain',
        'covariance_mode': 'separate',  # Explicitly documents that common_cov was NOT used
        'subject_output': str(output_dir),
        'fsaverage_dir': str(fsaverage_dir)
    }
    with open(output_dir / 'pipeline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Epoch source estimation complete: {output_dir}")
    log.info(f"{'='*60}\n")
    return metadata


def execute_source_estimation_epochs(
    project_base: Path,
    subject_id: str,
    task: str,
    epochs_file_path: str,
    fsaverage_dir: Path,
    reg: float = 0.05,
    n_jobs: int = 1,
    verbose: bool = False,
    noise_cov_method: str = 'shrunk',
    baseline_tmin: Optional[float] = None,
    baseline_tmax: Optional[float] = 0.0,
    data_cov_tmin: Optional[float] = None,
    data_cov_tmax: Optional[float] = None,
    compute_noise_cov: bool = True,
) -> Dict:
    """High-level orchestrator for epoch-based LCMV source estimation."""
    project_base = Path(project_base)
    package_dir = Path(lcmv_xtra.__file__).parent
    gpsc_full_path = package_dir / 'data' / 'bel_280' / 'ghw280_from_egig.gpsc'

    if not gpsc_full_path.exists():
        raise FileNotFoundError(f"Bundled .gpsc file not found: {gpsc_full_path}")

    epochs_full_path = project_base / epochs_file_path
    output_dir = project_base / 'derivatives' / 'lcmv' / f'{subject_id}_{task}_epochs'

    epochs, ch_pos = load_epochs(
        epochs_file_path=epochs_full_path, gpsc_file_path=gpsc_full_path,
        subject_id=subject_id, logger=None
    )

    return lcmv_beamformer_epochs(
        epochs=epochs, ch_pos=ch_pos, fsaverage_dir=fsaverage_dir, output_dir=output_dir,
        subject_id=subject_id, task=task,
        reg=reg, n_jobs=n_jobs, verbose=verbose,
        noise_cov_method=noise_cov_method,
        baseline_tmin=baseline_tmin,
        baseline_tmax=baseline_tmax,
        data_cov_tmin=data_cov_tmin,
        data_cov_tmax=data_cov_tmax,
        compute_noise_cov=compute_noise_cov,
    )
