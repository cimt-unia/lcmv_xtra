# lcmv_xtra/source_freq_band.py
import mne
import json
import logging
import numpy as np
from pathlib import Path

# Import shared utilities from the main source_estimation module
from .source_estimation import (
    load_subject, 
    validate_fsaverage, 
    _run_coregistration, 
    _setup_logger
)
import lcmv_xtra

def lcmv_beamformer_band(
    input,
    ch_pos,
    fsaverage_dir,
    output_dir,
    subject_id,
    task,
    band_name,
    fmin,
    fmax,
    reg=0.05,
    n_jobs=1,
    apply_to_broadband=False,
    verbose=False
):
    """
    Run LCMV source estimation with a band-specific covariance matrix.
    
    Parameters:
        ... (standard parameters) ...
        band_name: str, name of the frequency band (e.g., 'theta', 'low_beta')
        fmin: float, lower frequency bound (Hz)
        fmax: float, upper frequency bound (Hz)
        apply_to_broadband: bool, if True, applies the band-optimized spatial 
                            filter to the broadband raw data. If False (default), 
                            applies it to the bandpass-filtered data to yield
                            a pure band-limited source time course.
    """
    fsaverage_dir = Path(fsaverage_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log = _setup_logger(subject_id, f"{task}_{band_name}", output_dir, verbose)
    log.info(f"{'='*60}")
    log.info(f"Band-Limited LCMV: {subject_id} - {task} [{band_name}: {fmin}-{fmax} Hz]")
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

    # Bandpass filter for covariance estimation
    log.info(f"Filtering data for covariance estimation ({fmin} - {fmax} Hz)...")
    raw_filt = input.copy().filter(
        l_freq=fmin, h_freq=fmax, picks='eeg', 
        n_jobs=n_jobs, verbose=False
    )

    # LCMV beamformer
    log.info("Computing band-limited covariance and LCMV filters...")
    cov = mne.compute_raw_covariance(
        raw_filt, method='oas', picks='eeg', rank=None, n_jobs=n_jobs, verbose=False
    )
    filters = mne.beamformer.make_lcmv(
        info=raw_filt.info, forward=fwd, data_cov=cov, noise_cov=cov, reg=reg,
        pick_ori='max-power', weight_norm='unit-noise-gain',
        reduce_rank=True, rank=None, verbose=False
    )
    
    # Apply filter
    target_raw = input if apply_to_broadband else raw_filt
    target_desc = "broadband" if apply_to_broadband else f"{fmin}-{fmax} Hz filtered"
    log.info(f"Applying LCMV beamformer to {target_desc} data...")
    
    stc = mne.beamformer.apply_lcmv_raw(raw=target_raw, filters=filters)
    
    # Save with band-specific filename
    stc_file = output_dir / f'source_estimate_LCMV_{band_name}.h5'
    stc.save(stc_file, ftype='h5', overwrite=True)
    log.info(f"Saved source estimate: {stc.data.shape[0]} × {stc.data.shape[1]}")

    # Metadata
    metadata = {
        'subject_id': subject_id,
        'task': task,
        'band_name': band_name,
        'fmin': fmin,
        'fmax': fmax,
        'apply_to_broadband': apply_to_broadband,
        'sfreq_hz': float(input.info['sfreq']),
        'duration_min': float(input.n_times / input.info['sfreq'] / 60),
        'n_sources': int(stc.data.shape[0]),
        'n_timepoints': int(stc.data.shape[1]),
        'coreg_mean_error_mm': float(coreg_errors['mean']),
        'regularization': reg,
        'subject_output': str(output_dir),
        'fsaverage_dir': str(fsaverage_dir)
    }
    with open(output_dir / f'pipeline_metadata_{band_name}.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    log.info(f"Band-limited source estimation complete: {output_dir}")
    log.info(f"{'='*60}\n")
    
    return metadata


def execute_source_estimation_band(
    project_base,
    subject_id,
    task,
    ica_file_path,
    fsaverage_dir,
    band_name,
    fmin,
    fmax,
    reg=0.05,
    n_jobs=1,
    apply_to_broadband=False,
    verbose=False
):
    """
    High-level orchestrator for band-limited LCMV source estimation.
    """
    project_base = Path(project_base)
    package_dir = Path(lcmv_xtra.__file__).parent
    gpsc_full_path = package_dir / 'data' / 'bel_280' / 'ghw280_from_egig.gpsc'
    
    if not gpsc_full_path.exists():
        raise FileNotFoundError(f"Bundled .gpsc file not found: {gpsc_full_path}")
    
    ica_full_path = project_base / ica_file_path
    # Save in the same subject directory; filenames will differentiate the bands
    output_dir = project_base / 'derivatives' / 'lcmv' / f'{subject_id}_{task}'

    # Load data without logging (logging handled in lcmv_beamformer_band)
    raw, ch_pos = load_subject(
        ica_file_path=ica_full_path,
        gpsc_file_path=gpsc_full_path,
        subject_id=subject_id,
        logger=None
    )
    
    return lcmv_beamformer_band(
        input=raw,
        ch_pos=ch_pos,
        fsaverage_dir=fsaverage_dir,
        output_dir=output_dir,
        subject_id=subject_id,
        task=task,
        band_name=band_name,
        fmin=fmin,
        fmax=fmax,
        reg=reg,
        n_jobs=n_jobs,
        apply_to_broadband=apply_to_broadband,
        verbose=verbose
    )
