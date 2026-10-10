# High-Density EEG Preprocessing Framework for LCMV Source Reconstruction

This guide explains how to prepare BEL 280-channel EEG data for the `lcmv_xtra` source reconstruction library using `xeeg_kit` and `autoreject`. 

> **Note on Paths:** This tutorial is designed to be generic. In every code block, locate the `USER CONFIGURATION` section and update the `PROJECT_ROOT` and file paths to match your local directory structure.

<br>

### Introduction

```text
Raw Data ──► [Step 1: Continuous Clean] ──► [Step 2: Visual QA & Interpolation] ──► [Step 3: Epoching] ──► [Step 4: Epoch Repair] ──► Analysis-Ready Epochs
```

| Step | Input | Operation | Output |
| :--- | :--- | :--- | :--- |
| **1. Continuous Clean** | Raw Data (EDF, MFF, FIF, etc.) | MEEGKit (ASR+STAR+SNS) + ICLabel | `*_eeg_raw_eeg.fif` |
| **2. Visual QA & Interp** | Cleaned FIF | Manual bad channel review + Spherical Spline Interpolation | `*_interp_eeg.fif` |
| **3. Epoching** | Interpolated FIF | Event-aligned segmentation (no baseline correction) | `*_epo.fif` |
| **4. Epoch Repair** | Raw Epochs | RANSAC (global) + AutoReject (trial-specific) | `*_epo_clean.fif` |

<br>

## 1. Continuous EEG Preprocessing

Runs the native `xeeg_kit` pipeline on the full continuous recording. This applies ASR (Artifact Subspace Reconstruction), STAR (Sparse Time-Artifact Removal), SNS (Sensor Noise Suppression), and ICLabel. 

**Critical Design Choices:** 
1. **Universal File Loading:** We use `mne.io.read_raw()`, which automatically detects and routes to the correct parser based on the file extension (EDF, EGI/MFF, FIF, BrainVision, etc.).
2. **Deferred Interpolation:** `interpolate_bads` is set to `False` in both MEEGKit and ICLabel. This preserves the raw topography of bad channels so you can visually inspect them in Step 2 before committing to spherical spline interpolation.

```python
"""Continuous EEG Preprocessing using the native xeeg_kit entry point."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import mne
from xeeg_kit import preprocess_bel_trials

# ============================================================================
# USER CONFIGURATION (UPDATE THESE PATHS)
# ============================================================================

# Define your project root directory
PROJECT_ROOT = Path("/path/to/your/project")

# Path to your raw data file. 
# Supports: .edf, .mff (EGI), .fif, .vhdr (BrainVision), .set (EEGLAB), .bdf
RAW_DATA_PATH = PROJECT_ROOT / "raw_data" / "sub-01_task.edf"

# Output directory for cleaned data and reports
OUTPUT_DIR = PROJECT_ROOT / "derivatives" / "sub-01" / "continuous_clean"

# Region-specific line noise frequency (50.0 for Europe/Asia, 60.0 for Americas)
NOTCH_FREQ: float = 60.0

# MEEGKit parameters (ASR + STAR + SNS)
MEEGKIT_PARAMS: Dict[str, Any] = {
    "highpass_filter": 1.0,          # Required for ASR/ICA stability
    "low_pass_filter": 100.0,
    "notch_filter_freq": NOTCH_FREQ,       
    "mad_threshold": 10.0,           # Conservative bad channel detection
    "min_amplitude_uv": 0.5,
    "asr_cutoff": 3.5,               # Standard cutoff for high-density EEG
    "star_thresh": 2.5,              # Eccentricity threshold for transient artifacts
    "sns_neighbors": 8,              # Spatial neighbors for sensor noise suppression
    "drop_cz": True,                 # Drop hardware reference before CAR
    "interpolate_bads": False,       # CRITICAL: Defer interpolation to Step 2
    "generate_report": True,
}

# ICLabel refinement parameters
ICALABEL_PARAMS: Dict[str, Any] = {
    "mad_threshold": 50.0,           # Relaxed threshold; MEEGKit already cleaned gross artifacts
    "min_amplitude_uv": 0.5,
    "n_components": 0.99,            # Explain 99% of variance
    "random_state": 42,
    "interpolate_bads": False,       # CRITICAL: Defer interpolation to Step 2
    "generate_report": True,
}

# ============================================================================

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def process_continuous_raw(raw_path: Path, output_dir: Path) -> None:
    """Convert raw data to FIF, then run the full native xeeg_kit pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    # Step 1: Convert to FIF (preprocess_bel_trials requires FIF input)
    staging_dir = output_dir / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staging_fif = staging_dir / f"{raw_path.stem}_eeg_raw.fif" 
    
    if not staging_fif.exists():
        logger.info("Converting %s to FIF for library compatibility...", raw_path.suffix)
        
        # Universal MNE loader: Automatically routes to the correct parser 
        # based on file extension. This replaces the need for format-specific calls:
        # mne.io.read_raw_edf(...)   -> EDF
        # mne.io.read_raw_egi(...)   -> MFF / EGI
        # mne.io.read_raw_fif(...)   -> FIF
        # mne.io.read_raw_brainvision(...) -> BrainVision
        raw_data = mne.io.read_raw(str(raw_path), preload=False, verbose="WARNING")
        
        raw_data.save(str(staging_fif), overwrite=True, verbose="WARNING")
        logger.info("Saved raw FIF: %s", staging_fif)
    else:
        logger.info("Staging FIF already exists, skipping conversion.")

    # Step 2: Run Native Pipeline
    logger.info("Running preprocess_bel_trials...")
    results = preprocess_bel_trials(
        data_dir=staging_dir,
        output_dir=output_dir,
        meegkit_params=MEEGKIT_PARAMS,
        icalabel_params=ICALABEL_PARAMS,
        pattern=staging_fif.name,
        overwrite=True,
        verbose=True,
    )

    logger.info("Pipeline complete. %d file(s) processed.", len(results))

if __name__ == "__main__":
    process_continuous_raw(raw_path=RAW_DATA_PATH, output_dir=OUTPUT_DIR)
```

<br>

## 2. Visual QA and Bad Channel Interpolation

**Run this in a Jupyter Notebook after Step 1.** 
This step loads the continuously cleaned data, allows you to visually identify remaining bad channels (e.g., jaw/EMG channels that ASR/ICLabel might have missed), interpolates them using spherical splines, and applies the final Common Average Reference (CAR).

```python
# Jupyter Inspection & Interpolation Block
import mne
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", force=True)
logger = logging.getLogger(__name__)

# Ensure interactive backend is active in Jupyter
%matplotlib widget  

# ============================================================================
# USER CONFIGURATION (UPDATE THESE PATHS)
# ============================================================================
PROJECT_ROOT = Path("/path/to/your/project")
CLEANED_FIF = PROJECT_ROOT / "derivatives" / "sub-01" / "continuous_clean" / "sub-01_task_eeg_raw_eeg.fif"
# ============================================================================

if not CLEANED_FIF.exists():
    raise FileNotFoundError(f"Run Step 1 first. Missing: {CLEANED_FIF}")

raw_clean = mne.io.read_raw_fif(str(CLEANED_FIF), preload=True, verbose="WARNING")

logger.info("Channels: %d | Duration: %.1fs | Bads: %s", 
            len(raw_clean.ch_names), raw_clean.n_times / raw_clean.info['sfreq'], raw_clean.info['bads'])

# Open interactive plot. Scroll through channels, click on names to mark as bad.
raw_clean.plot(n_channels=50, title="CONTINUOUS CLEAN QA (Click channel names to mark bad)", verbose="WARNING");
```

```python
# %% Run this cell AFTER closing the interactive plot window above

# Add specific non-neural channels programmatically (e.g., BEL jaw/EMG channels)
MANUAL_BADS = ["E52", "E42", "E43", "E280", "E275", "E276"] 

# Merge interactive selections + manual additions
updated_bads = sorted(set(raw_clean.info["bads"] + MANUAL_BADS))
raw_clean.info["bads"] = updated_bads
logger.info("Final bad channels (%d): %s", len(updated_bads), updated_bads)

# Interpolate and Re-reference
logger.info("Interpolating %d bad channels...", len(updated_bads))
raw_clean.interpolate_bads(reset_bads=True)
raw_clean.set_eeg_reference("average", projection=False, verbose=False)
logger.info("Re-applied CAR after interpolation.")

# Save to NEW file
INTERPOLATED_FIF = CLEANED_FIF.with_stem(CLEANED_FIF.stem + "_interp_eeg")
raw_clean.save(str(INTERPOLATED_FIF), overwrite=True, verbose="WARNING")
logger.info("Saved interpolated data: %s", INTERPOLATED_FIF.name)
```

<br>

## 3. Event-Aligned Epoching

Segments the interpolated continuous data into trials aligned to your cognitive event (e.g., `choice_press_ttl`). 

**Critical Design Choice:** `baseline=None` is used here. Baseline correction is mathematically deferred to the LCMV beamformer's noise covariance matrix computation, which provides superior spatial whitening compared to simple time-domain subtraction.

```python
"""Create epochs from interpolated continuous EEG."""
from __future__ import annotations

import logging
from pathlib import Path
import mne
import numpy as np
import pandas as pd

# ============================================================================
# USER CONFIGURATION (UPDATE THESE PATHS)
# ============================================================================
PROJECT_ROOT = Path("/path/to/your/project")

INTERPOLATED_FIF = PROJECT_ROOT / "derivatives" / "sub-01" / "continuous_clean" / "sub-01_task_eeg_raw_eeg_interp_eeg.fif"
VALIDATION_CSV = PROJECT_ROOT / "logs" / "sub-01_validation.csv" # Your event onsets
OUTPUT_EPO = PROJECT_ROOT / "derivatives" / "sub-01" / "eeg_epochs" / "sub-01_epo.fif"

EPOCH_TMIN: float = -2.5
EPOCH_TMAX: float = 2.5
ALIGNMENT_COLUMN: str = "choice_press_ttl" # Column in CSV with event times in seconds
# ============================================================================

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def create_epochs() -> None:
    raw = mne.io.read_raw_fif(str(INTERPOLATED_FIF), preload=True, verbose="WARNING")
    df = pd.read_csv(VALIDATION_CSV)
    df_valid = df[df["valid_choice_trial"] == True].copy()
    
    sfreq = raw.info["sfreq"]
    onset_samples = np.round(df_valid[ALIGNMENT_COLUMN].to_numpy() * sfreq).astype(int)
    events_array = np.column_stack([onset_samples, np.zeros(len(onset_samples), dtype=int), np.ones(len(onset_samples), dtype=int)])

    # baseline=None is critical for downstream LCMV noise covariance whitening
    epochs = mne.Epochs(raw, events_array, event_id=1, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, 
                        baseline=None, preload=True, verbose="WARNING")

    OUTPUT_EPO.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(str(OUTPUT_EPO), overwrite=True, verbose="WARNING")
    logger.info("Saved %d epochs → %s", len(epochs), OUTPUT_EPO.name)

if __name__ == "__main__":
    create_epochs()
```

<br>

## 4. Epoch-Level Artifact Rejection (AutoReject)

Loads the raw epochs and applies a two-stage cleaning process optimized for 280-channel high-density arrays:
1. **RANSAC:** Identifies globally unpredictable channels across the entire recording and interpolates them.
2. **AutoReject:** Uses Bayesian optimization to find the optimal per-channel thresholds. It repairs transient artifacts (e.g., a single eye blink in one trial) via local interpolation, and only drops the entire trial if the artifact is too widespread to repair.

```python
"""Epoch-Level Artifact Rejection and Repair using Autoreject."""
from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
from autoreject import AutoReject, Ransac

# ============================================================================
# USER CONFIGURATION (UPDATE THESE PATHS)
# ============================================================================
PROJECT_ROOT = Path("/path/to/your/project")

INPUT_EPOCHS = PROJECT_ROOT / "derivatives" / "sub-01" / "eeg_epochs" / "sub-01_epo.fif"
OUTPUT_EPOCHS = PROJECT_ROOT / "derivatives" / "sub-01" / "eeg_epochs" / "sub-01_epo_clean.fif"
OUTPUT_LOG = PROJECT_ROOT / "derivatives" / "sub-01" / "eeg_epochs" / "sub-01_reject_log.npz"

# AutoReject Parameters (Optimized for 280-channel High-Density EEG)
# Max 32 channels interpolated per trial (~11.4% of 280; safe for spherical splines)
N_INTERPOLATE: np.ndarray = np.array([1, 4, 16, 32])
CONSENSUS: np.ndarray = np.linspace(0.1, 1.0, 10)
# ============================================================================

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def process_epochs() -> None:
    epochs = mne.read_epochs(str(INPUT_EPOCHS), preload=True, verbose="WARNING")
    logger.info("Loaded %d epochs, %d channels.", len(epochs), len(epochs.ch_names))

    # 1. RANSAC (Global bad channel detection)
    logger.info("Running RANSAC...")
    ransac = Ransac(n_resample=50, min_channels=0.25, min_corr=0.75, unbroken_time=0.4, n_jobs=-1, verbose=True)
    ransac.fit(epochs)

    if ransac.bad_chs_:
        logger.info("RANSAC identified %d globally bad channels.", len(ransac.bad_chs_))
        epochs.info["bads"] = ransac.bad_chs_
        epochs.interpolate_bads(reset_bads=True)

    # 2. AutoReject (Trial-specific transient artifact repair)
    logger.info("Fitting AutoReject...")
    ar = AutoReject(n_interpolate=N_INTERPOLATE, consensus=CONSENSUS, random_state=42, n_jobs=-1, verbose=True)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    
    logger.info("AutoReject complete. Retained %d / %d epochs.", len(epochs_clean), len(epochs))

    # 3. Save Outputs
    epochs_clean.save(str(OUTPUT_EPOCHS), overwrite=True, verbose="WARNING")
    reject_log.save(str(OUTPUT_LOG), overwrite=True)
    
    # 4. QC Plots
    fig_log = reject_log.plot(orientation="horizontal", show=False)
    fig_log.savefig(str(OUTPUT_LOG.with_suffix(".png")), dpi=150, bbox_inches="tight")
    plt.close(fig_log)

if __name__ == "__main__":
    process_epochs()
```

<br>

## 5. Final Epoch Inspection

**Run this in a Jupyter Notebook after Step 4.**
Opens the MNE interactive epoch browser. Use `Page Down` / `Page Up` to scroll through all 280 channels (50 at a time). Verify that the AutoReject repairs look clean and that the readiness potential / movement execution signals are intact.

```python
# %% Jupyter Epoch Inspection Block
import mne
from pathlib import Path

%matplotlib widget  

# ============================================================================
# USER CONFIGURATION (UPDATE THESE PATHS)
# ============================================================================
PROJECT_ROOT = Path("/path/to/your/project")
CLEAN_EPO = PROJECT_ROOT / "derivatives" / "sub-01" / "eeg_epochs" / "sub-01_epo_clean.fif"
# ============================================================================

epochs_clean = mne.read_epochs(str(CLEAN_EPO), preload=True, verbose="WARNING")

# n_channels=50 shows 50 channels at a time. 
# Use Page Down/Up to scroll through all 280 channels.
# Use - / + to adjust amplitude scaling.
fig_browser = epochs_clean.plot(
    n_epochs=1, 
    n_channels=50, 
    events=False, 
    title="Interactive Epoch Inspector (Post-AutoReject)",
)
```

<br>

### Integration with `lcmv_xtra`

The output of Step 4 (`sub-01_epo_clean.fif`) is the exact input expected by the refactored `lcmv_xtra.source_estimation_epochs` library. 

Because you deferred baseline correction (`baseline=None` in Step 3) and avoided double-whitening (by keeping MEEGKit/ICLabel interpolation separate from LCMV noise covariance), the beamformer will correctly use your manually defined time windows (e.g., Noise: `[-2.5, -2.0]s`, Data: `[-2.0, 2.0]s`) to construct unbiased spatial filters and extract single-trial time series for your M1 and STN ROIs.
