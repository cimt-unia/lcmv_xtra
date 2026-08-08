# lcmv_xtra/epoch_tensor.py
"""
Epoched LCMV Source Estimation
==============================
Splits raw EEG into fixed-duration segments, runs independent LCMV
beamforming on each segment (per-epoch covariance), extracts time 
courses at user-specified MNI coordinates, and assembles an epoched tensor.

This is distinct from:
  - Fixed LCMV (one filter for entire recording)
  - Post-hoc epoching (continuous LCMV then split time courses via lcmv_stats)

Here: raw → split → per-epoch LCMV → per-epoch STC → tensor.
"""

import json
import logging
import numpy as np
import mne
from pathlib import Path
from typing import Dict, List, Optional, Literal

from .source_estimation import (
    load_subject,
    validate_fsaverage,
    _run_coregistration,
    _setup_logger,
)
from .custom_tensor import extract_custom_roi_time_courses

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_EPOCH_SEC: float = 10.0
DEFAULT_OVERLAP_SEC: float = 0.0
DEFAULT_REG: float = 0.05
DEFAULT_TARGET_SFREQ: float = 250.0
MIN_TN_RATIO: float = 8.0  # Warning threshold for BEL 280


def _split_raw_into_epochs(
    raw: mne.io.Raw,
    epoch_duration_sec: float,
    overlap_sec: float,
) -> List[mne.io.Raw]:
    """Split continuous Raw into fixed-duration segments."""
    sfreq = raw.info["sfreq"]
    n_times = raw.n_times
    epoch_samples = int(np.round(epoch_duration_sec * sfreq))
    step_samples = max(1, int(np.round((epoch_duration_sec - overlap_sec) * sfreq)))

    if epoch_samples > n_times:
        logger.warning(
            f"Epoch ({epoch_samples} samples) > data ({n_times} samples). "
            "Returning single epoch with all data."
        )
        return [raw.copy()]

    epochs: List[mne.io.Raw] = []
    start = 0
    while start + epoch_samples <= n_times:
        tmin = start / sfreq
        tmax = (start + epoch_samples) / sfreq
        segment = raw.copy().crop(tmin=tmin, tmax=tmax, include_last=True)
        epochs.append(segment)
        start += step_samples

    logger.info(
        f"Split into {len(epochs)} epochs "
        f"({epoch_duration_sec}s, overlap={overlap_sec}s, sfreq={sfreq}Hz)"
    )
    return epochs


def _compute_forward_once(
    raw: mne.io.Raw,
    ch_pos: Dict,
    fsaverage_dir: Path,
    output_dir: Path,
    log: logging.Logger,
) -> mne.Forward:
    """Compute coregistration and forward model ONCE for all epochs."""
    bem_file, src_file = validate_fsaverage(fsaverage_dir)

    log.info("Running coregistration (once for all epochs)...")
    trans_file = output_dir / "fsaverage-trans.fif"
    trans, _ = _run_coregistration(
        raw, ch_pos, "fsaverage", fsaverage_dir, trans_file, log
    )

    log.info("Computing forward solution (once)...")
    fwd_file = output_dir / "fsaverage-vol-eeg-fwd.fif"
    bem = mne.read_bem_solution(bem_file)
    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=mne.read_source_spaces(src_file), 
        bem=bem, eeg=True, mindist=5.0, n_jobs=1,
    )
    mne.write_forward_solution(fwd_file, fwd, overwrite=True)
    return fwd


def _run_lcmv_single_epoch(
    epoch_raw: mne.io.Raw,
    fwd: mne.Forward,
    epoch_index: int,
    output_dir: Path,
    reg: float,
    log: logging.Logger,
) -> mne.SourceEstimate:
    """Compute covariance, LCMV filter, and STC for ONE epoch."""
    n_channels = len(mne.pick_types(epoch_raw.info, eeg=True, exclude="bads"))
    n_samples = epoch_raw.n_times
    tn_ratio = n_samples / max(n_channels, 1)

    if tn_ratio < MIN_TN_RATIO:
        log.warning(
            f"Epoch {epoch_index:03d}: T/N={tn_ratio:.1f} < {MIN_TN_RATIO}. "
            "Covariance may be unreliable."
        )

    cov = mne.compute_raw_covariance(
        epoch_raw, method="oas", picks="eeg", rank=None,
        n_jobs=1, verbose=False,
    )

    filters = mne.beamformer.make_lcmv(
        info=epoch_raw.info, forward=fwd, data_cov=cov, noise_cov=cov,
        reg=reg, pick_ori="max-power", weight_norm="unit-noise-gain",
        reduce_rank=True, rank=None, verbose=False,
    )

    stc = mne.beamformer.apply_lcmv_raw(raw=epoch_raw, filters=filters)

    # Save per-epoch STC for debugging/inspection
    epoch_label = f"epoch{epoch_index:03d}"
    stc_file = output_dir / f"source_estimate_LCMV_{epoch_label}.h5"
    stc.save(stc_file, ftype="h5", overwrite=True, verbose=False)
    
    log.info(
        f"  {epoch_label}: {stc.data.shape[0]} src × "
        f"{stc.data.shape[1]} samp (T/N={tn_ratio:.1f})"
    )
    return stc


def execute_epoch_tensor(
    project_base: Path | str,
    subject_id: str,
    task: str,
    ica_file_path: str | Path,
    fsaverage_dir: Path | str,
    roi_coordinates: Dict[str, List[float]],
    epoch_duration_sec: float = DEFAULT_EPOCH_SEC,
    overlap_sec: float = DEFAULT_OVERLAP_SEC,
    radius_mm: float = 5.0,
    mode: Literal["sphere", "single"] = "sphere",
    reg: float = DEFAULT_REG,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
    verbose: bool = False,
) -> Path:
    """
    Run epoched LCMV source estimation for a single subject.
    
    Splits raw EEG into fixed-duration segments, runs independent LCMV
    beamforming on each segment, extracts time courses at user-specified
    MNI coordinates, and assembles an epoched tensor.
    """
    if not roi_coordinates:
        raise ValueError("roi_coordinates must contain at least one ROI.")

    project_base = Path(project_base)
    fsaverage_dir = Path(fsaverage_dir)
    ica_path = project_base / ica_file_path
    
    # Create subject output directory
    output_dir = project_base / "derivatives" / "lcmv" / f"{subject_id}_{task}_epoched"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log = _setup_logger(subject_id, f"{task}_epoched", output_dir, verbose)

    log.info("=" * 60)
    log.info(f"EPOCHED LCMV: {subject_id} - {task}")
    log.info(f"  Epoch: {epoch_duration_sec}s | Overlap: {overlap_sec}s")
    log.info(f"  ROIs: {list(roi_coordinates.keys())}")
    log.info(f"  Mode: {mode} | Radius: {radius_mm}mm")
    log.info("=" * 60)

    # 1. Load subject
    import lcmv_xtra
    package_dir = Path(lcmv_xtra.__file__).parent
    gpsc_path = package_dir / "data" / "bel_280" / "ghw280_from_egig.gpsc"

    raw, ch_pos = load_subject(
        ica_file_path=ica_path,
        gpsc_file_path=gpsc_path,
        subject_id=subject_id,
        logger=log,
    )

    # Downsample before splitting to match target_sfreq
    if raw.info["sfreq"] > target_sfreq:
        log.info(f"Downsampling: {raw.info['sfreq']:.0f}Hz → {target_sfreq:.0f}Hz")
        raw = raw.copy().resample(target_sfreq, npad="auto")

    # 2. Forward model (computed once)
    fwd = _compute_forward_once(raw, ch_pos, fsaverage_dir, output_dir, log)

    _, src_file = validate_fsaverage(fsaverage_dir)
    src = mne.read_source_spaces(src_file)

    # 3. Split into epochs
    epochs_raw = _split_raw_into_epochs(raw, epoch_duration_sec, overlap_sec)

    # 4. LCMV per epoch + ROI extraction
    epoch_time_courses: List[np.ndarray] = []
    roi_names: Optional[List[str]] = None

    log.info(f"Processing {len(epochs_raw)} epochs...")
    for idx, epoch_raw in enumerate(epochs_raw):
        stc = _run_lcmv_single_epoch(epoch_raw, fwd, idx, output_dir, reg, log)
        
        tc, names = extract_custom_roi_time_courses(
            stc=stc, src=src,
            roi_coordinates=roi_coordinates,
            radius_mm=radius_mm,
            mode=mode,
            log=log,
        )
        epoch_time_courses.append(tc)
        if roi_names is None:
            roi_names = names

    if not epoch_time_courses:
        raise RuntimeError("No epochs were successfully processed.")

    # 5. Assemble tensor
    min_samples = min(tc.shape[1] for tc in epoch_time_courses)
    truncated = [tc[:, :min_samples] for tc in epoch_time_courses]
    stacked = np.stack(truncated, axis=0)  # (n_epochs, n_rois, n_samples)

    sfreq = raw.info["sfreq"]
    tensor_file = output_dir / f"{subject_id}_{task}_epoched_tensor.npz"
    np.savez_compressed(
        tensor_file,
        data=stacked,
        roi_names=np.array(roi_names),
        sfreq=sfreq,
        epoch_duration_sec=epoch_duration_sec,
        overlap_sec=overlap_sec,
        n_epochs=len(epoch_time_courses),
        is_epoched=True,
    )
    log.info(f"Saved epoched tensor: {stacked.shape} → {tensor_file}")

    # 6. Save metadata
    metadata = {
        "subject_id": subject_id,
        "task": task,
        "strategy": "epoched_lcmv",
        "n_epochs": len(epochs_raw),
        "epoch_duration_sec": epoch_duration_sec,
        "overlap_sec": overlap_sec,
        "sfreq_hz": float(sfreq),
        "n_rois": len(roi_names),
        "roi_names": roi_names,
        "roi_coordinates": roi_coordinates,
        "radius_mm": radius_mm,
        "mode": mode,
        "regularization": reg,
        "subject_output": str(output_dir),
        "fsaverage_dir": str(fsaverage_dir),
    }
    meta_path = output_dir / "pipeline_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata saved: {meta_path}")

    log.info("=" * 60)
    log.info("Epoched LCMV complete.")
    log.info("=" * 60)
    
    return tensor_file

'''
# Usage Example
from lcmv_xtra import execute_epoch_tensor
from pathlib import Path

mni_rois_coords = {
    "R1": [10.958, 5.563, -4.948],
    "L1": [-15.983, 6.077, 0.027],
    "STN-L": [-11.89, -14.51, -6.40],
}

tensor_path = execute_epoch_tensor(
    project_base="/xtra",
    subject_id="Sb02",
    task="gain",
    ica_file_path="eeg_cleaned.fif",
    fsaverage_dir="fs",
    roi_coordinates=mni_rois_coords,
    epoch_duration_sec=10.0,
    overlap_sec=0.0,
    verbose=True,
)
'''
