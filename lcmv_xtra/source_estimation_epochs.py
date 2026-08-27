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


def load_subject(ica_file_path: Path, gpsc_file_path: Path, 
                 subject_id: Optional[str] = None, 
                 logger: Optional[logging.Logger] = None) -> Tuple[mne.io.Raw, Dict]:
    """Load and preprocess subject data (identical to continuous version)."""
    log = logger or logging.getLogger(__name__)
    ica_file = Path(ica_file_path)
    gpsc_file = Path(gpsc_file_path)
    
    if not ica_file.exists():
        raise FileNotFoundError(f"ICA file not found: {ica_file}")
    if not gpsc_file.exists():
        raise FileNotFoundError(f"GPSC file not found: {gpsc_file}")

    raw = mne.io.read_raw_fif(ica_file, preload=True)
    sfreq = raw.info['sfreq']
    duration_min = raw.n_times / sfreq / 60
    subject_str = f" ({subject_id})" if subject_id else ""
    log.info(f"Loaded data: {duration_min:.1f}min @ {sfreq}Hz{subject_str}")

    existing_channels = set(raw.info['ch_names'])
    valid_channel_map = {old: new for old, new in _BEL_CHANNEL_MAP.items() if old in existing_channels}
    if valid_channel_map:
        raw.rename_channels(valid_channel_map)

    channels = parse_gpsc(gpsc_file)
    if not channels:
        raise ValueError("No valid channels found in .gpsc file")
    
    gpsc_array = np.array([ch[1:4] for ch in channels])
    mean_pos = np.mean(gpsc_array, axis=0)
    channels_normalized = [
        (ch[0], ch[1] - mean_pos[0], ch[2] - mean_pos[1], ch[3] - mean_pos[2]) 
        for ch in channels
    ]
    ch_pos = {ch[0]: np.array(ch[1:4]) / 1000.0 for ch in channels_normalized}
    
    missing_fids = [fid for fid in _REQUIRED_FIDUCIALS if fid not in ch_pos]
    if missing_fids:
        raise ValueError(f"Missing required fiducials: {missing_fids}")

    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos, nasion=ch_pos['FidNz'], lpa=ch_pos['FidT9'], rpa=ch_pos['FidT10'], coord_frame='head'
    )
    raw.set_montage(montage, on_missing='warn')
    raw = raw.pick(['eeg', 'stim'], exclude=raw.info['bads'])

    has_avg_ref = any(p['desc'] == 'average' for p in raw.info['projs'])
    if not has_avg_ref:
        raw.set_eeg_reference('average', projection=True)
    if not raw.proj:
        raw.apply_proj()

    log.info("Preprocessing complete")
    return raw, ch_pos


def validate_fsaverage(subjects_dir: Path) -> Tuple[Path, Path]:
    """Validate fsaverage resources."""
    subjects_dir = Path(subjects_dir)
    fsaverage_dir = subjects_dir / 'fsaverage'
    bem_file = fsaverage_dir / 'bem' / 'fsaverage-5120-5120-5120-bem-sol.fif'
    src_file = subjects_dir / 'fsaverage-vol-5mm-src.fif'
    
    if not (bem_file.exists() and src_file.exists()):
        raise FileNotFoundError(f"Missing fsaverage files in {subjects_dir}")
    return bem_file, src_file


def _run_coregistration(raw: mne.io.Raw, ch_pos: Dict, subject: str, 
                        subjects_dir: Path, trans_file: Path, 
                        logger: logging.Logger) -> Tuple[mne.transforms.Transform, Dict]:
    """Run enhanced coregistration with ICP and outlier removal."""
    log = logger or logging.getLogger(__name__)
    coreg = mne.coreg.Coregistration(
        raw.info, subject=subject, subjects_dir=subjects_dir,
        fiducials={'nasion': ch_pos['FidNz'], 'lpa': ch_pos['FidT9'], 'rpa': ch_pos['FidT10']}
    )
    coreg.fit_fiducials(verbose=False)
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=False)
    
    dists = coreg.compute_dig_mri_distances()
    if np.sum(dists > 5.0/1000) > 0:
        coreg.omit_head_shape_points(distance=5.0/1000)
        
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
    raw: mne.io.Raw,
    ch_pos: Dict,
    fsaverage_dir: Path,
    output_dir: Path,
    subject_id: str,
    task: str,
    epoch_duration: float = 2.0,
    reg: float = 0.05,
    n_jobs: int = 1,
    verbose: bool = False,
    noise_cov_method: str = 'shrunk',
    baseline_tmin: Optional[float] = None,
    baseline_tmax: float = 0.1
) -> Dict:
    """
    Run epoch-based LCMV source estimation with proper noise/data covariance separation.

    Cuts continuous data into non-overlapping epochs, computes separate noise
    (from baseline) and data (full epoch) covariances, and applies whitened
    LCMV filters via apply_lcmv_epochs.

    Parameters
    ----------
    noise_cov_method : str
        Estimator for noise covariance ('shrunk', 'oas', 'empirical').
        'shrunk' (Ledoit-Wolf) is recommended for short baseline windows.
    baseline_tmin : float | None
        Start of baseline window for noise covariance (seconds relative to epoch tmin=0).
        None defaults to 0.0 (epoch onset).
    baseline_tmax : float
        End of baseline window for noise covariance (seconds relative to epoch tmin=0).
        Must be < epoch_duration.
    """
    fsaverage_dir = Path(fsaverage_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log = _setup_logger(subject_id, task, output_dir, verbose)
    log.info(f"{'='*60}")
    log.info(f"LCMV Epoch Source Estimation: {subject_id} - {task}")
    log.info(f"{'='*60}")

    # Validate baseline window
    if baseline_tmax >= epoch_duration:
        raise ValueError(
            f"baseline_tmax ({baseline_tmax}s) must be < epoch_duration ({epoch_duration}s)"
        )
    noise_tmin = baseline_tmin if baseline_tmin is not None else 0.0
    if noise_tmin >= baseline_tmax:
        raise ValueError(
            f"baseline_tmin ({noise_tmin}s) must be < baseline_tmax ({baseline_tmax}s)"
        )

    # 1. Coregistration & Forward (computed once on full info)
    bem_file, src_file = validate_fsaverage(fsaverage_dir)
    trans_file = output_dir / 'fsaverage-trans.fif'
    trans, coreg_errors = _run_coregistration(raw, ch_pos, 'fsaverage', fsaverage_dir, trans_file, log)
    
    src = mne.read_source_spaces(src_file)
    fwd_file = output_dir / 'fsaverage-vol-eeg-fwd.fif'
    bem = mne.read_bem_solution(bem_file)
    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=n_jobs
    )
    mne.write_forward_solution(fwd_file, fwd, overwrite=True)

    # 2. Re-epoch the concatenated continuous data
    sfreq = raw.info['sfreq']
    log.info(f"Cutting continuous data into {epoch_duration}s non-overlapping epochs...")
    events = mne.make_fixed_length_events(raw, duration=epoch_duration, overlap=0.0)
    
    # tmax is inclusive; subtract 1 sample to get exact epoch_duration
    tmax = epoch_duration - (1.0 / sfreq)
    epochs = mne.Epochs(
        raw, events, event_id=None, tmin=0.0, tmax=tmax,
        baseline=None, preload=True, proj=True
    )
    log.info(f"Created {len(epochs)} epochs.")
    
    if len(epochs) == 0:
        raise RuntimeError("No epochs created. Check epoch_duration vs data length.")

    # 3. Compute SEPARATE noise and data covariances
    epochs_eeg = epochs.copy().pick('eeg')

    # Noise covariance from baseline period within each epoch
    log.info(
        f"Computing NOISE covariance from baseline [{noise_tmin:.3f}, {baseline_tmax:.3f}]s "
        f"using method='{noise_cov_method}'..."
    )
    noise_cov = mne.compute_covariance(
        epochs_eeg, tmin=noise_tmin, tmax=baseline_tmax,
        method=noise_cov_method, rank=None, n_jobs=n_jobs, verbose=False
    )

    # Data covariance from full epoch (signal + noise)
    log.info("Computing DATA covariance from full epochs using method='oas'...")
    data_cov = mne.compute_covariance(
        epochs_eeg, tmin=0.0, tmax=tmax,
        method='oas', rank=None, n_jobs=n_jobs, verbose=False
    )

    # Log rank information for debugging rank mismatches
    data_rank = mne.compute_rank(data_cov, info=epochs_eeg.info)
    noise_rank = mne.compute_rank(noise_cov, info=epochs_eeg.info)
    log.info(f"Data covariance rank: {data_rank}")
    log.info(f"Noise covariance rank: {noise_rank}")

    # 4. Make LCMV filters with proper noise covariance whitening
    log.info("Computing LCMV filters with separate noise/data covariance...")
    filters = mne.beamformer.make_lcmv(
        info=epochs.info, forward=fwd,
        data_cov=data_cov,
        noise_cov=noise_cov,
        reg=reg,
        pick_ori='max-power',
        weight_norm='unit-noise-gain',
        reduce_rank=True,
        rank=None,
        verbose=False
    )

    # 5. Apply LCMV to epochs (whitening handled internally via filters['whitener'])
    log.info("Applying LCMV beamformer to epochs...")
    stcs: List[mne.SourceEstimate] = mne.beamformer.apply_lcmv_epochs(
        epochs=epochs, filters=filters
    )
    
    # 6. Save each epoch STC individually
    log.info(f"Saving {len(stcs)} epoch source estimates...")
    for i, stc in enumerate(stcs):
        stc_file = output_dir / f'source_estimate_LCMV_epoch_{i:03d}.h5'
        stc.save(stc_file, ftype='h5', overwrite=True)

    # Metadata with noise covariance details
    metadata = {
        'subject_id': subject_id,
        'task': task,
        'sfreq_hz': float(sfreq),
        'epoch_duration_sec': float(epoch_duration),
        'n_epochs': len(stcs),
        'n_sources': int(stcs[0].data.shape[0]),
        'n_timepoints_per_epoch': int(stcs[0].data.shape[1]),
        'coreg_mean_error_mm': float(coreg_errors['mean']),
        'regularization': reg,
        'data_covariance_method': 'epoch_averaged_oas',
        'noise_covariance_method': noise_cov_method,
        'noise_baseline_window': [float(noise_tmin), float(baseline_tmax)],
        'data_rank': data_rank,
        'noise_rank': noise_rank,
        'weight_normalization': 'unit-noise-gain',
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
    ica_file_path: str,
    fsaverage_dir: Path,
    epoch_duration: float = 2.0,
    reg: float = 0.05,
    n_jobs: int = 1,
    verbose: bool = False,
    noise_cov_method: str = 'shrunk',
    baseline_tmin: Optional[float] = None,
    baseline_tmax: float = 0.1
) -> Dict:
    """High-level orchestrator for epoch-based LCMV source estimation."""
    project_base = Path(project_base)
    package_dir = Path(lcmv_xtra.__file__).parent
    gpsc_full_path = package_dir / 'data' / 'bel_280' / 'ghw280_from_egig.gpsc'
    
    if not gpsc_full_path.exists():
        raise FileNotFoundError(f"Bundled .gpsc file not found: {gpsc_full_path}")
    
    ica_full_path = project_base / ica_file_path
    output_dir = project_base / 'derivatives' / 'lcmv' / f'{subject_id}_{task}_epochs'

    raw, ch_pos = load_subject(
        ica_file_path=ica_full_path, gpsc_file_path=gpsc_full_path,
        subject_id=subject_id, logger=None
    )
    
    return lcmv_beamformer_epochs(
        raw=raw, ch_pos=ch_pos, fsaverage_dir=fsaverage_dir, output_dir=output_dir,
        subject_id=subject_id, task=task, epoch_duration=epoch_duration,
        reg=reg, n_jobs=n_jobs, verbose=verbose,
        noise_cov_method=noise_cov_method,
        baseline_tmin=baseline_tmin,
        baseline_tmax=baseline_tmax
    )


'''
import lcmv_xtra as lx
from pathlib import Path

# Define your paths
PROJECT_BASE = Path("/path/to/project")
FS_DIR = Path("/path/to/fsaverage")

# Run epoch-based source estimation with proper noise covariance
metadata = lx.execute_source_estimation_epochs(
    project_base=PROJECT_BASE,
    subject_id="sub-01",
    task="rest",
    ica_file_path="sub-01/ses-01/eeg/sub-01_rest_cleaned.fif",
    fsaverage_dir=FS_DIR,
    epoch_duration=2.0,
    reg=0.05,
    n_jobs=1,
    verbose=True,
    noise_cov_method='shrunk',     # Ledoit-Wolf shrinkage for short baselines
    baseline_tmin=None,            # Baseline starts at epoch onset (0.0s)
    baseline_tmax=0.1              # 100ms baseline window for noise estimation
)
'''
