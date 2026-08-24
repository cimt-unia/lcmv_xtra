# lcmv_xtra/dl_tensor_epochs.py
"""
Neural Beamformer Epoch Tensor Assembly
=======================================
Runs continuous neural beamformer per subject via dl_beamformer,
then performs post-hoc epoching of the (448, T) source estimates.

Output: 4D Tensor (Subjects, 448 ROIs, Epochs, Time_per_epoch).
ROI subsetting via MNI coordinates is external (use lookup_mni_coordinate).
"""

import os
import logging
import numpy as np
import pandas as pd
import mne
from pathlib import Path
from scipy import signal
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Relative imports within the library
from .dl_beamformer import execute_source_estimation_atlas_pytorch

logger = logging.getLogger(__name__)


def epoch_continuous_data(data: np.ndarray, sfreq: float, epoch_duration: float) -> np.ndarray:
    """Cuts continuous (N_rois, T) data into (N_epochs, N_rois, Time_per_epoch)."""
    n_rois, n_timepoints = data.shape
    samples_per_epoch = int(epoch_duration * sfreq)

    if samples_per_epoch == 0:
        raise ValueError("Epoch duration too short for sampling rate.")

    n_epochs = n_timepoints // samples_per_epoch
    valid_timepoints = n_epochs * samples_per_epoch

    data_trimmed = data[:, :valid_timepoints]
    data_epoched = data_trimmed.reshape(n_rois, n_epochs, samples_per_epoch)
    # Transpose to (N_epochs, N_rois, Time_per_epoch) for stacking convention
    return np.transpose(data_epoched, (1, 0, 2))


def _process_single_subject(args: Tuple) -> Dict:
    """Run neural beamformer on continuous data, then epoch the output."""
    (sid, fif_path, task_name, project_base, fs_dir,
     epoch_duration, nn_epochs, reg, n_jobs_inner, verbose) = args

    try:
        # 1. Run continuous neural beamformer
        metadata = execute_source_estimation_atlas_pytorch(
            project_base=str(project_base),
            subject_id=sid,
            task=task_name,
            ica_file_path=str(fif_path),
            fsaverage_dir=str(fs_dir),
            reg=reg,
            n_jobs=n_jobs_inner,
            nn_epochs=nn_epochs,
            verbose=verbose,
        )

        subject_output = Path(metadata["subject_output"])
        sfreq = metadata["sfreq_hz"]

        # 2. Load continuous neural STC (448, T)
        stc_npy = subject_output / "source_estimate_neural.npy"
        stc_h5 = subject_output / "source_estimate_LCMV.h5"

        if stc_npy.exists():
            stc_data = np.load(stc_npy)
        elif stc_h5.exists():
            stc = mne.read_source_estimate(str(stc_h5))
            stc_data = stc.data
        else:
            raise FileNotFoundError(f"No neural STC found for {sid} in {subject_output}")

        # 3. Epoch ALL 448 ROIs → (N_epochs, 448, Time_per_epoch)
        epoch_tc = epoch_continuous_data(stc_data, sfreq, epoch_duration)

        return {
            "subject_id": sid,
            "epoch_time_courses": epoch_tc,
            "sfreq": sfreq,
            "n_epochs": epoch_tc.shape[0],
            "success": True,
        }
    except Exception as e:
        logger.error("Failed %s: %s", sid, e)
        return {"subject_id": sid, "success": False, "error": str(e)}


def save_dl_tensor_epochs(
    all_subject_data: List[Dict],
    task_name: str,
    output_dir: Path,
    target_sfreq: float = 250.0,
) -> Path:
    """Stack into 4D array: (Subjects, 448, Epochs, Time_per_epoch)."""
    if not all_subject_data:
        raise ValueError("No successful subjects to stack.")

    output_dir.mkdir(parents=True, exist_ok=True)

    resampled_data = []
    subject_ids = []

    for d in all_subject_data:
        etc = d["epoch_time_courses"]  # (n_epochs, 448, time)
        current_sfreq = d["sfreq"]

        if abs(current_sfreq - target_sfreq) > 0.5:
            up = int(target_sfreq)
            down = int(round(current_sfreq))
            gcd = np.gcd(up, down)
            etc_res = signal.resample_poly(etc, up // gcd, down // gcd, axis=2)
            resampled_data.append(etc_res)
        else:
            resampled_data.append(etc)
        subject_ids.append(d["subject_id"])

    min_epochs = min(arr.shape[0] for arr in resampled_data)
    min_time = min(arr.shape[2] for arr in resampled_data)

    stacked = np.stack([
        arr[:min_epochs, :, :min_time] for arr in resampled_data
    ])  # (n_subjects, min_epochs, 448, min_time)

    # Reorder → (Subjects, 448, Epochs, Time)
    stacked = np.transpose(stacked, (0, 2, 1, 3))

    output_path = output_dir / f"study_{task_name}_dl_epochs.npz"
    np.savez_compressed(
        output_path,
        data=stacked,
        subject_ids=np.array(subject_ids),
        sfreq=target_sfreq,
        n_epochs=min_epochs,
        n_rois=stacked.shape[1],
        beamformer_type="neural_cimt_448",
    )

    logger.info(
        "Saved 4D Neural Epoch Tensor %s (Subjects, ROIs, Epochs, Time) "
        "at %.1f Hz to %s", stacked.shape, target_sfreq, output_path
    )
    return output_path


def assemble_dl_tensor_epochs(
    data_index: pd.DataFrame,
    fs_dir: Path,
    output_dir: Path,
    task_name: str = "study",
    project_base: Optional[Path] = None,
    epoch_duration: float = 2.5,
    nn_epochs: int = 5500,
    reg: float = 0.05,
    n_jobs: int = 1,
    n_jobs_inner: int = 1,
    verbose: bool = False,
    target_sfreq: float = 250.0,
) -> Optional[Path]:
    """
    Assemble 4D epoch tensor using the Neural Beamformer.

    Runs continuous neural beamformer per subject, then post-hoc epoching.
    Output contains ALL 448 CIMT ROIs. Subset externally via ROI indices.

    Parameters
    ----------
    data_index : pd.DataFrame
        Must contain 'subject_id' and 'fif_path' columns.
    fs_dir : Path
        Path to fsaverage directory.
    output_dir : Path
        Where to save the final .npz tensor.
    task_name : str
        Label for the output file.
    project_base : Path, optional
        Base project directory. Defaults to cwd.
    epoch_duration : float
        Length of each non-overlapping epoch in seconds.
    nn_epochs : int
        Max training epochs for neural beamformer.
    reg : float
        LCMV regularization parameter.
    n_jobs : int
        Number of parallel subjects. Default 1 (GPU-safe).
    n_jobs_inner : int
        Parallel workers within each subject's forward/covariance computation.
    verbose : bool
        Enable verbose logging.
    target_sfreq : float
        Target sampling rate for the output tensor.
    """
    if data_index.empty:
        return None
    if project_base is None:
        project_base = Path.cwd()

    tasks = [
        (
            row["subject_id"],
            Path(row["fif_path"]),
            task_name,
            project_base,
            fs_dir,
            epoch_duration,
            nn_epochs,
            reg,
            n_jobs_inner,
            verbose,
        )
        for _, row in data_index.iterrows()
    ]

    # GPU safety: force n_jobs=1 when CUDA is available
    if n_jobs == -1 or n_jobs > 1:
        try:
            import torch
            if torch.cuda.is_available() and n_jobs != 1:
                logger.warning(
                    "CUDA detected. Forcing n_jobs=1 to prevent GPU OOM "
                    "during parallel neural beamforming."
                )
                n_jobs = 1
        except ImportError:
            pass

    if n_jobs == -1:
        n_jobs = os.cpu_count()

    logger.info("Processing %d subjects with n_jobs=%d (Neural Beamformer)...", len(tasks), n_jobs)

    all_subject_data = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        future_to_sid = {
            executor.submit(_process_single_subject, task): task[0]
            for task in tasks
        }
        for future in as_completed(future_to_sid):
            result = future.result()
            if result.get("success"):
                all_subject_data.append(result)
            else:
                logger.error("Failed %s: %s", result["subject_id"], result.get("error", "Unknown"))

    if all_subject_data:
        return save_dl_tensor_epochs(
            all_subject_data, task_name, output_dir, target_sfreq=target_sfreq
        )
    return None
