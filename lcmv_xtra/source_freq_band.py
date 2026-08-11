# lcmv_xtra/source_freq_band.py
"""DICS frequency-band source estimation pipeline.

Mirrors the structure of source_estimation.py but uses DICS beamforming
in the frequency domain instead of LCMV in the time domain. Optimized for
replicating LFP spectral content at specific frequency bands.
"""

import mne
import json
import logging
import lcmv_xtra
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd

from .source_estimation import load_subject, validate_fsaverage, _run_coregistration, _setup_logger

# MNE logging
mne.set_log_level('warning')


def dics_beamformer(
    input: mne.io.Raw,
    ch_pos: dict,
    fsaverage_dir: Path,
    output_dir: Path,
    subject_id: str,
    task: str,
    freq_band: Tuple[float, float] = (4.0, 8.0),
    n_freq_bins: int = 9,
    reg: float = 0.001,
    real_filter: bool = False,
    weight_norm: Optional[str] = None,
    depth: float = 1.0,
    pick_ori: str = "max-power",
    reduce_rank: bool = False,
    epoch_duration_sec: float = 7.5,
    n_jobs: int = 1,
    verbose: bool = False,
) -> dict:
    """
    Run full DICS frequency-band source estimation pipeline.

    Computes cross-spectral density at specified frequency bins, builds
    DICS spatial filters, and produces a source power map averaged across
    the frequency band. Output is structurally compatible with LCMV atlas
    extraction tools after frequency averaging.

    Parameters
    ----------
    input : Raw
        Preprocessed raw EEG data.
    ch_pos : dict
        Channel positions from GPSC file.
    fsaverage_dir : Path
        Directory containing fsaverage BEM and source space.
    output_dir : Path
        Output directory for results.
    subject_id : str
        Subject identifier.
    task : str
        Task/condition name.
    freq_band : tuple of (float, float)
        Frequency band of interest (fmin, fmax) in Hz.
    n_freq_bins : int
        Number of frequency bins within the band for CSD computation.
    reg : float
        Regularization for CSD inversion. Default 0.001 optimized for
        EEG-only DICS with complex CSD.
    real_filter : bool
        If True, use only real part of CSD. False preserves complex
        phase information, which improves deep source localization.
    weight_norm : str or None
        Weight normalization. None works best for DICS with self-noise CSD.
    depth : float or None
        Depth weighting. 1.0 is DICS default; None disables (LCMV equivalent).
    pick_ori : str
        Orientation constraint. 'max-power' recommended for power maps.
    reduce_rank : bool
        Rank reduction. 
    epoch_duration_sec : float
        Duration of synthetic epochs for CSD computation.
    n_jobs : int
        Parallel jobs for forward model computation.
    verbose : bool
        Enable console logging.

    Returns
    -------
    metadata : dict
        Pipeline metadata including paths for downstream atlas extraction.
    """
    fsaverage_dir = Path(fsaverage_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log = _setup_logger(subject_id, task, output_dir, verbose)
    log.info(f"{'='*60}")
    log.info(f"DICS Frequency-Band Source Estimation: {subject_id} - {task}")
    log.info(f"Band: {freq_band[0]}-{freq_band[1]} Hz ({n_freq_bins} bins)")
    log.info(f"{'='*60}")

    # Validate and get resource paths
    bem_file, src_file = validate_fsaverage(fsaverage_dir)

    # Coregistration
    log.info("Running coregistration...")
    trans_file = output_dir / 'fsaverage-trans.fif'
    trans, coreg_errors = _run_coregistration(
        input, ch_pos, 'fsaverage', fsaverage_dir, trans_file, log
    )

    # Source space
    log.info("Loading source space...")
    src = mne.read_source_spaces(src_file)
    n_active = len(src[0]['vertno'])
    log.info(f"Source space: {n_active} active sources")

    # Forward solution
    log.info("Computing forward solution...")
    fwd_file = output_dir / 'fsaverage-vol-eeg-fwd.fif'
    bem = mne.read_bem_solution(bem_file)
    fwd = mne.make_forward_solution(
        input.info, trans=trans, src=src, bem=bem,
        eeg=True, mindist=5.0, n_jobs=n_jobs
    )
    mne.write_forward_solution(fwd_file, fwd, overwrite=True)

    # Create synthetic epochs for CSD computation
    sfreq = input.info['sfreq']
    epoch_samples = int(epoch_duration_sec * sfreq)
    n_epochs = input.n_times // epoch_samples

    if n_epochs < 2:
        raise ValueError(
            f"Not enough data for CSD: {n_epochs} epochs of {epoch_duration_sec}s. "
            f"Need at least 2 epochs."
        )

    events = np.column_stack([
        np.arange(n_epochs) * epoch_samples,
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])
    epochs = mne.Epochs(
        input, events, event_id=1,
        tmin=0.0, tmax=epoch_duration_sec,
        baseline=None, preload=True
    )
    log.info(f"Created {n_epochs} synthetic epochs ({epoch_duration_sec}s each)")

    # Compute CSD at frequency bins within the band
    freqs = np.linspace(freq_band[0], freq_band[1], n_freq_bins)
    log.info(f"Computing CSD at {freqs.tolist()} Hz...")
    csd_signal = csd_morlet(
        epochs, freqs, tmin=0.0, tmax=epoch_duration_sec, decim=1
    )
    # Use signal CSD as noise CSD (required for EEG without separate baseline)
    csd_noise = csd_signal.copy()
    log.info("Using signal CSD as noise CSD (EEG-only, no separate baseline)")

    # Build DICS spatial filters
    log.info("Computing DICS spatial filters...")
    log.info(f"  reg={reg}, real_filter={real_filter}, weight_norm={weight_norm}")
    log.info(f"  depth={depth}, pick_ori={pick_ori}, reduce_rank={reduce_rank}")

    # Enforce valid parameter combinations
    if reduce_rank and pick_ori == "vector":
        log.warning("reduce_rank=True with pick_ori='vector' can be unstable; proceeding with caution.")

    filters = make_dics(
        epochs.info, fwd, csd_signal,
        noise_csd=csd_noise,
        reg=reg,
        pick_ori=pick_ori,
        reduce_rank=reduce_rank,
        real_filter=real_filter,
        weight_norm=weight_norm,
        depth=depth,
        verbose=False,
    )

    # Apply DICS to get source power per frequency
    log.info("Applying DICS filters to CSD...")
    stc, stc_freqs = apply_dics_csd(csd_signal, filters)
    log.info(f"Source power map: {stc.data.shape[0]} sources × {len(stc_freqs)} frequencies")

    # Average across frequency bins → single band-power map
    avg_stc = stc.copy()
    avg_stc.data = np.mean(stc.data, axis=1, keepdims=True)

    # Save both per-frequency and averaged maps
    stc_file = output_dir / 'source_estimate_DICS_per_freq.h5'
    stc.save(stc_file, ftype='h5', overwrite=True)

    avg_stc_file = output_dir / 'source_estimate_DICS_avg_band.h5'
    avg_stc.save(avg_stc_file, ftype='h5', overwrite=True)
    log.info(f"Saved averaged band power map: {avg_stc.data.shape[0]} × {avg_stc.data.shape[1]}")

    # Metadata (compatible with atlas extraction pipeline)
    metadata = {
        'subject_id': subject_id,
        'task': task,
        'method': 'DICS',
        'sfreq_hz': float(input.info['sfreq']),
        'duration_min': float(input.n_times / input.info['sfreq'] / 60),
        'n_sources': int(avg_stc.data.shape[0]),
        'n_timepoints': int(avg_stc.data.shape[1]),  # Added for lcmv_xtra atlas extraction compatibility
        'n_frequencies': len(stc_freqs),
        'freq_band_hz': list(freq_band),
        'freq_bins_hz': [float(f) for f in stc_freqs],
        'n_epochs': n_epochs,
        'epoch_duration_sec': epoch_duration_sec,
        'coreg_mean_error_mm': float(coreg_errors['mean']),
        'dics_params': {
            'reg': reg,
            'real_filter': real_filter,
            'weight_norm': weight_norm,
            'depth': depth,
            'pick_ori': pick_ori,
            'reduce_rank': reduce_rank,
        },
        'subject_output': str(output_dir),
        'fsaverage_dir': str(fsaverage_dir),
        'avg_band_stc_file': str(avg_stc_file),
        'per_freq_stc_file': str(stc_file),
    }
    with open(output_dir / 'pipeline_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    log.info(f"DICS frequency-band source estimation complete: {output_dir}")
    log.info(f"{'='*60}\n")

    return metadata


def execute_source_estimation_band(
    project_base: Path,
    subject_id: str,
    task: str,
    ica_file_path: str,
    fsaverage_dir: Path,
    freq_band: Tuple[float, float] = (4.0, 8.0),
    n_freq_bins: int = 9,
    reg: float = 0.001,
    real_filter: bool = False,
    weight_norm: Optional[str] = None,
    depth: float = 1.0,
    pick_ori: str = "max-power",
    reduce_rank: bool = False,
    epoch_duration_sec: float = 7.5,
    n_jobs: int = 1,
    verbose: bool = False,
) -> dict:
    """
    High-level orchestrator for DICS frequency-band source estimation.

    Constructs standard paths within a BIDS-like project and delegates
    to dics_beamformer for the actual computation. Mirrors the interface
    of execute_source_estimation() from source_estimation.py.

    Parameters
    ----------
    project_base : Path
        Root project directory.
    subject_id : str
        Subject identifier.
    task : str
        Task/condition name.
    ica_file_path : str
        Relative path to ICA-cleaned FIF file (relative to project_base).
    fsaverage_dir : Path
        Directory containing fsaverage BEM and source space.
    freq_band : tuple of (float, float)
        Frequency band of interest in Hz.
    n_freq_bins : int
        Number of frequency bins for CSD.
    reg : float
        DICS regularization.
    real_filter : bool
        Use real-only CSD (False preserves phase for deep sources).
    weight_norm : str or None
        Weight normalization scheme.
    depth : float or None
        Depth weighting value.
    pick_ori : str
        Orientation constraint.
    reduce_rank : bool
        Enable rank reduction.
    epoch_duration_sec : float
        Synthetic epoch length for CSD computation.
    n_jobs : int
        Parallel jobs.
    verbose : bool
        Enable console logging.

    Returns
    -------
    metadata : dict
        Pipeline metadata including output paths.
    """
    project_base = Path(project_base)
    package_dir = Path(lcmv_xtra.__file__).parent
    gpsc_full_path = package_dir / 'data' / 'bel_280' / 'ghw280_from_egig.gpsc'

    if not gpsc_full_path.exists():
        raise FileNotFoundError(f"Bundled .gpsc file not found: {gpsc_full_path}")

    ica_full_path = project_base / ica_file_path
    output_dir = project_base / 'derivatives' / 'lcmv' / f'{subject_id}_{task}_dics'

    # Load data without logging (logging handled in dics_beamformer)
    raw, ch_pos = load_subject(
        ica_file_path=ica_full_path,
        gpsc_file_path=gpsc_full_path,
        subject_id=subject_id,
        logger=None
    )

    return dics_beamformer(
        input=raw,
        ch_pos=ch_pos,
        fsaverage_dir=fsaverage_dir,
        output_dir=output_dir,
        subject_id=subject_id,
        task=task,
        freq_band=freq_band,
        n_freq_bins=n_freq_bins,
        reg=reg,
        real_filter=real_filter,
        weight_norm=weight_norm,
        depth=depth,
        pick_ori=pick_ori,
        reduce_rank=reduce_rank,
        epoch_duration_sec=epoch_duration_sec,
        n_jobs=n_jobs,
        verbose=verbose,
    )
