# lcmv_xtra/atlas_tensor.py
"""
CIMT Atlas-Constrained Tensor Assembly
========================================
Source estimation → Direct 448-ROI tensor (no post-hoc extraction).

Uses execute_source_estimation_atlas() which reduces the forward model
to 448 CIMT ROI columns BEFORE make_lcmv(). The output STC is already
(448, T), so no cimt_extraction() step is needed.

This is distinct from:
  - tensor.py: Standard LCMV + post-hoc cimt_extraction()
  - custom_tensor.py: Standard LCMV + user-defined MNI sphere extraction

Usage:
    import lcmv_xtra as lx
    df = lx.scan_eeg_paths(CLEAN_DIR, pattern="*_rest_cleaned.fif")
    lx.assemble_atlas_tensor(df, FS_DIR, OUTPUT_DIR, task_name="rest")
"""

import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from concurrent.futures import ProcessPoolExecutor, as_completed

from lcmv_xtra.source_estimation_atlas import execute_source_estimation_atlas

logger = logging.getLogger(__name__)

DEFAULT_TARGET_SFREQ = 250.0


# =============================================================================
# SINGLE SUBJECT PROCESSING
# =============================================================================

def _process_single_subject_atlas(args):
    """
    Worker function for parallel CIMT-constrained source estimation.

    Unlike tensor.py's _process_single_subject, this does NOT call
    cimt_extraction() because execute_source_estimation_atlas() already
    produces (448, T) output directly.
    """
    sid, fif_path, task_name, project_base, fs_dir, reg, n_jobs_inner, verbose = args

    try:
        metadata = execute_source_estimation_atlas(
            project_base=project_base,
            subject_id=sid,
            task=task_name,
            ica_file_path=str(fif_path),
            fsaverage_dir=fs_dir,
            reg=reg,
            n_jobs=n_jobs_inner,
            verbose=verbose,
        )

        # Load the (448, T) source estimate directly — no extraction needed
        import mne
        stc_file = Path(metadata['subject_output']) / "source_estimate_LCMV.h5"
        stc = mne.read_source_estimate(str(stc_file))

        # Verify shape
        assert stc.data.shape[0] == 448, \
            f"Expected 448 ROIs, got {stc.data.shape[0]} for {sid}"

        return {
            "subject_id": sid,
            "time_courses": stc.data.astype(np.float32),  # (448, T)
            "sfreq": metadata['sfreq_hz'],
            "n_timepoints": metadata['n_timepoints'],
            "coreg_error_mm": metadata['coreg_mean_error_mm'],
            "success": True,
        }

    except Exception as e:
        logger.error(f"⚠️ Failed {sid}: {e}")
        return {"subject_id": sid, "success": False, "error": str(e)}


# =============================================================================
# RESAMPLING & STACKING
# =============================================================================

def _resample_time_course(time_course, current_sfreq, target_sfreq):
    """Resample a single time course array to target frequency."""
    if abs(current_sfreq - target_sfreq) < 0.5:
        return time_course

    up = int(target_sfreq)
    down = int(round(current_sfreq))

    if up == int(target_sfreq) and down == int(round(current_sfreq)):
        gcd = np.gcd(up, down)
        return signal.resample_poly(time_course, up // gcd, down // gcd, axis=1)
    else:
        duration_sec = time_course.shape[1] / current_sfreq
        n_samples_new = int(np.round(duration_sec * target_sfreq))
        return signal.resample(time_course, n_samples_new, axis=1)


def save_atlas_study_tensor(
    all_subject_data: list,
    task_name: str,
    output_dir: Path,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
) -> Path:
    """
    Stack CIMT atlas data into (subjects, 448, time) tensor.

    Resamples all subjects to target_sfreq and truncates to shortest
    time series for uniform shape.
    """
    if not all_subject_data:
        raise ValueError("No successful subject data to stack.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resample each subject to target frequency
    resampled_data = []
    for d in all_subject_data:
        tc = d['time_courses']  # (448, T)
        tc_resampled = _resample_time_course(tc, d['sfreq'], target_sfreq)
        resampled_data.append(tc_resampled)

    # Truncate to shortest time series
    min_t = min(arr.shape[1] for arr in resampled_data)
    stacked_data = np.stack([arr[:, :min_t] for arr in resampled_data])

    subject_ids = np.array([d['subject_id'] for d in all_subject_data])

    # Load canonical CIMT ROI names from bundled labels
    import lcmv_xtra
    labels_path = Path(lcmv_xtra.__file__).parent / 'data' / 'cimt_atlas' / 'cimt_atlas_labels.csv'
    roi_names = pd.read_csv(labels_path)['roi_name'].tolist()

    output_path = output_dir / f"study_{task_name}_cimt.npz"
    np.savez_compressed(
        output_path,
        data=stacked_data,
        subject_ids=subject_ids,
        roi_names=np.array(roi_names),
        sfreq=target_sfreq,
        n_rois=448,
        atlas="CIMT_448",
    )

    logger.info(
        f"✅ Saved CIMT Atlas Tensor: {stacked_data.shape} "
        f"@ {target_sfreq} Hz → {output_path}"
    )
    return output_path


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def assemble_atlas_tensor(
    data_index: pd.DataFrame,
    fs_dir: Path,
    output_dir: Path,
    task_name: str = "study",
    project_base: Path = None,
    reg: float = 0.05,
    n_jobs: int = -1,
    n_jobs_inner: int = 1,
    verbose: bool = False,
    target_sfreq: float = DEFAULT_TARGET_SFREQ,
) -> Path:
    """
    CIMT-constrained source estimation → 448-ROI study tensor.

    Drop-in replacement for assemble_tensor() that uses atlas-before-inverse
    source estimation. No post-hoc cimt_extraction() is performed.

    Parameters
    ----------
    data_index : pd.DataFrame
        Must contain 'subject_id' and 'fif_path' columns.
    fs_dir : Path
        Directory containing fsaverage/ and fsaverage-vol-5mm-src.fif.
    output_dir : Path
        Where to save the final .npz tensor.
    task_name : str
        Label for the output file (e.g., 'rest', 'gain').
    project_base : Path, optional
        Root of BIDS-like project. Defaults to cwd.
    reg : float
        LCMV regularization parameter (default 0.05).
    n_jobs : int
        Number of parallel subjects (-1 = all CPUs).
    n_jobs_inner : int
        Parallel jobs per subject for forward solution (default 1).
        Set >1 only if n_jobs=1 to avoid oversubscription.
    verbose : bool
        Enable console logging for each subject.
    target_sfreq : float
        Target sampling frequency for the output tensor.

    Returns
    -------
    Path to saved .npz file, or None if all subjects failed.
    """
    if data_index.empty:
        logger.warning("Empty data_index provided.")
        return None

    if project_base is None:
        project_base = Path.cwd()

    # Build task list
    tasks = [
        (
            row['subject_id'],
            Path(row['fif_path']),
            task_name,
            project_base,
            fs_dir,
            reg,
            n_jobs_inner,
            verbose,
        )
        for _, row in data_index.iterrows()
    ]

    if n_jobs == -1:
        n_jobs = os.cpu_count()

    logger.info(f">>> Processing {len(tasks)} subjects with {n_jobs} workers...")
    logger.info(f"    Backend: CIMT atlas-before-inverse (448 ROIs)")
    logger.info(f"    Regularization: {reg}, Inner jobs: {n_jobs_inner}")

    all_subject_data = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_sid = {
            executor.submit(_process_single_subject_atlas, task): task[0]
            for task in tasks
        }
        for future in as_completed(future_to_sid):
            result = future.result()
            if result.get('success'):
                all_subject_data.append(result)
                logger.info(
                    f"  ✅ {result['subject_id']}: "
                    f"{result['time_courses'].shape} @ {result['sfreq']}Hz "
                    f"(coreg={result['coreg_error_mm']:.2f}mm)"
                )
            else:
                logger.error(
                    f"  ❌ {result['subject_id']}: {result.get('error', 'Unknown error')}"
                )

    if not all_subject_data:
        logger.error("All subjects failed. No tensor produced.")
        return None

    logger.info(f">>> {len(all_subject_data)}/{len(tasks)} subjects succeeded.")

    return save_atlas_study_tensor(
        all_subject_data, task_name, output_dir, target_sfreq=target_sfreq
    )
