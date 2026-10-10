# High-Density EEG Preprocessing Framework

This guide explains how to prepare BEL 280-channel EEG data for the `lcmv_xtra` source reconstruction library using `xeeg_kit` and `autoreject`.


<br>

### Introduction


<br>

| Step | Input | Operation | Output |
| :--- | :--- | :--- | :--- |
| **1. Convert to FIF** | Raw Data (EDF, MFF, etc.) | Format-specific MNE reader → FIF | `*_eeg_raw.fif` |
| **2. Continuous Clean** | Staging FIF | MEEGKit (ASR+STAR+SNS) + ICLabel | `*_eeg_raw_eeg.fif` |
| **3. Visual QA & Interp** | Cleaned FIF | Manual bad channel review + Spherical Spline Interpolation | `*_interp_eeg.fif` |
| **4. Epoching** | Interpolated FIF | Event-aligned segmentation (no baseline correction) | `*_epo.fif` |
| **5. Epoch Repair** | Raw Epochs | RANSAC (global) + AutoReject (trial-specific) | `*_epo_clean.fif` |



<br>

## 1. Convert Raw Data to FIF

The `xeeg_kit` library strictly requires `.fif` files as input. Run this cell **once** to convert your raw data into the required staging format. Uncomment the reader that matches your file type.

```python
"""Convert raw EEG data to FIF for xeeg_kit compatibility."""
import logging
from pathlib import Path
import mne

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ============================================================================
# USER CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path("/path/to/your/project")
RAW_DATA_PATH = PROJECT_ROOT / "raw_data" / "sub-01_task.edf"
STAGING_DIR = PROJECT_ROOT / "derivatives" / "sub-01" / "continuous_clean" / "staging"
# ============================================================================

STAGING_DIR.mkdir(parents=True, exist_ok=True)
STAGING_FIF = STAGING_DIR / f"{RAW_DATA_PATH.stem}_eeg_raw.fif"

# Select the appropriate reader for your raw data format:
raw_data = mne.io.read_raw_edf(str(RAW_DATA_PATH), preload=False, verbose="WARNING")          # EDF TYPE

# raw_data = mne.io.read_raw_egi(str(RAW_DATA_PATH), preload=False, verbose="WARNING")        # MFF / EGI TYPE

# raw_data = mne.io.read_raw_fif(str(RAW_DATA_PATH), preload=False, verbose="WARNING")        # FIF TYPE

# raw_data = mne.io.read_raw_brainvision(str(RAW_DATA_PATH), preload=False, verbose="WARNING") # BrainVision TYPE

raw_data.save(str(STAGING_FIF), overwrite=True, verbose="WARNING")
logger.info("Converted %s → %s", RAW_DATA_PATH.name, STAGING_FIF.name)
```

<br>

## 2. Continuous EEG Preprocessing

Runs the native `xeeg_kit` pipeline on the staging FIF created in Step 1. This applies ASR, STAR, SNS, and ICLabel.

**Critical Design Choice:** `interpolate_bads` is set to `False` in both MEEGKit and ICLabel. This preserves the raw topography of bad channels so you can visually inspect them in Step 3 before committing to spherical spline interpolation.

```python
"""Continuous EEG Preprocessing using the native xeeg_kit entry point."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from xeeg_kit import preprocess_bel_trials

# ============================================================================
# USER CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path("/path/to/your/project")
STAGING_DIR = PROJECT_ROOT / "derivatives" / "sub-01" / "continuous_clean" / "staging"
OUTPUT_DIR = PROJECT_ROOT / "derivatives" / "sub-01" / "continuous_clean"
NOTCH_FREQ: float = 60.0  # 50.0 for Europe/Asia, 60.0 for Americas

MEEGKIT_PARAMS: Dict[str, Any] = {
    "highpass_filter": 1.0,
    "low_pass_filter": 100.0,
    "notch_filter_freq": NOTCH_FREQ,
    "mad_threshold": 10.0,
    "min_amplitude_uv": 0.5,
    "asr_cutoff": 3.5,
    "star_thresh": 2.5,
    "sns_neighbors": 8,
    "drop_cz": True,
    "interpolate_bads": False,       # CRITICAL: Defer interpolation to Step 3
    "generate_report": True,
}

ICALABEL_PARAMS: Dict[str, Any] = {
    "mad_threshold": 50.0,
    "min_amplitude_uv": 0.5,
    "n_components": 0.99,
    "random_state": 42,
    "interpolate_bads": False,       # CRITICAL: Defer interpolation to Step 3
    "generate_report": True,
}
# ============================================================================

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = preprocess_bel_trials(
    data_dir=STAGING_DIR,
    output_dir=OUTPUT_DIR,
    meegkit_params=MEEGKIT_PARAMS,
    icalabel_params=ICALABEL_PARAMS,
    pattern="*_eeg_raw.fif",
    overwrite=True,
    verbose=True,
)

logger.info("Pipeline complete. %d file(s) processed.", len(results))
```

<br>

## 3. Visual QA and Bad Channel Interpolation

**Run this in a Jupyter Notebook after Step 2.**
Load the continuously cleaned data, visually identify remaining bad channels, interpolate them using spherical splines, and apply the final Common Average Reference (CAR).

```python
# Cell 1: Interactive Inspection
import mne
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S", force=True)
logger = logging.getLogger(__name__)

%matplotlib widget

# ============================================================================
# USER CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path("/path/to/your/project")
CLEANED_FIF = PROJECT_ROOT / "derivatives" / "sub-01" / "continuous_clean" / "sub-01_task_eeg_raw_eeg.fif"
# ============================================================================

raw_clean = mne.io.read_raw_fif(str(CLEANED_FIF), preload=True, verbose="WARNING")

logger.info("Channels: %d | Duration: %.1fs | Bads: %s",
            len(raw_clean.ch_names), raw_clean.n_times / raw_clean.info['sfreq'], raw_clean.info['bads'])

# Scroll through channels, click on names to mark as bad
raw_clean.plot(n_channels=50, title="CONTINUOUS CLEAN QA (Click channel names to mark bad)", verbose="WARNING");
```

```python
# Cell 2: Interpolate & Save (run AFTER closing the plot above)

MANUAL_BADS = ["E52", "E42", "E43", "E280", "E275", "E276"]

updated_bads = sorted(set(raw_clean.info["bads"] + MANUAL_BADS))
raw_clean.info["bads"] = updated_bads
logger.info("Final bad channels (%d): %s", len(updated_bads), updated_bads)

raw_clean.interpolate_bads(reset_bads=True)
raw_clean.set_eeg_reference("average", projection=False, verbose=False)
logger.info("Interpolated %d channels and re-applied CAR.", len(updated_bads))

INTERPOLATED_FIF = CLEANED_FIF.with_stem(CLEANED_FIF.stem + "_interp_eeg")
raw_clean.save(str(INTERPOLATED_FIF), overwrite=True, verbose="WARNING")
logger.info("Saved interpolated data: %s", INTERPOLATED_FIF.name)
```

<br>

## 4. Event-Aligned Epoching

Segments the interpolated continuous data into trials aligned to your cognitive event.

**Critical Design Choice:** `baseline=None` is used here. Baseline correction is mathematically deferred to the LCMV beamformer's noise covariance matrix computation, which provides superior spatial whitening compared to simple time-domain subtraction.

```python
"""Create epochs from interpolated continuous EEG."""
import logging
from pathlib import Path
import mne
import numpy as np
import pandas as pd

# ============================================================================
# USER CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path("/path/to/your/project")

INTERPOLATED_FIF = PROJECT_ROOT / "derivatives" / "sub-01" / "continuous_clean" / "sub-01_task_eeg_raw_eeg_interp_eeg.fif"
VALIDATION_CSV = PROJECT_ROOT / "logs" / "sub-01_validation.csv"
OUTPUT_EPO = PROJECT_ROOT / "derivatives" / "sub-01" / "eeg_epochs" / "sub-01_epo.fif"

EPOCH_TMIN: float = -2.5
EPOCH_TMAX: float = 2.5
ALIGNMENT_COLUMN: str = "choice_press_ttl"
# ============================================================================

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

raw = mne.io.read_raw_fif(str(INTERPOLATED_FIF), preload=True, verbose="WARNING")
df = pd.read_csv(VALIDATION_CSV)
df_valid = df[df["valid_choice_trial"] == True].copy()

sfreq = raw.info["sfreq"]
onset_samples = np.round(df_valid[ALIGNMENT_COLUMN].to_numpy() * sfreq).astype(int)
events_array = np.column_stack([
    onset_samples,
    np.zeros(len(onset_samples), dtype=int),
    np.ones(len(onset_samples), dtype=int),
])

epochs = mne.Epochs(
    raw, events_array, event_id=1,
    tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
    baseline=None, preload=True, verbose="WARNING",
)

OUTPUT_EPO.parent.mkdir(parents=True, exist_ok=True)
epochs.save(str(OUTPUT_EPO), overwrite=True, verbose="WARNING")
logger.info("Saved %d epochs → %s", len(epochs), OUTPUT_EPO.name)
```

<br>

## 5. Epoch-Level Artifact Rejection (AutoReject)

Applies a two-stage cleaning process optimized for 280-channel high-density arrays:
1.  **RANSAC:** Identifies globally unpredictable channels and interpolates them.
2.  **AutoReject:** Uses Bayesian optimization to find optimal per-channel thresholds. Repairs transient artifacts via local interpolation; drops entire trials only when artifacts are too widespread to repair.

```python
"""Epoch-Level Artifact Rejection and Repair using Autoreject."""
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np
from autoreject import AutoReject, Ransac

# ============================================================================
# USER CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path("/path/to/your/project")

INPUT_EPOCHS = PROJECT_ROOT / "derivatives" / "sub-01" / "eeg_epochs" / "sub-01_epo.fif"
OUTPUT_EPOCHS = PROJECT_ROOT / "derivatives" / "sub-01" / "eeg_epochs" / "sub-01_epo_clean.fif"
OUTPUT_LOG = PROJECT_ROOT / "derivatives" / "sub-01" / "eeg_epochs" / "sub-01_reject_log.npz"

N_INTERPOLATE: np.ndarray = np.array([1, 4, 16, 32])
CONSENSUS: np.ndarray = np.linspace(0.1, 1.0, 10)
# ============================================================================

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

epochs = mne.read_epochs(str(INPUT_EPOCHS), preload=True, verbose="WARNING")
logger.info("Loaded %d epochs, %d channels.", len(epochs), len(epochs.ch_names))

ransac = Ransac(n_resample=50, min_channels=0.25, min_corr=0.75, unbroken_time=0.4, n_jobs=-1, verbose=True)
ransac.fit(epochs)

if ransac.bad_chs_:
    logger.info("RANSAC identified %d globally bad channels.", len(ransac.bad_chs_))
    epochs.info["bads"] = ransac.bad_chs_
    epochs.interpolate_bads(reset_bads=True)

ar = AutoReject(n_interpolate=N_INTERPOLATE, consensus=CONSENSUS, random_state=42, n_jobs=-1, verbose=True)
epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

logger.info("AutoReject complete. Retained %d / %d epochs.", len(epochs_clean), len(epochs))

epochs_clean.save(str(OUTPUT_EPOCHS), overwrite=True, verbose="WARNING")
reject_log.save(str(OUTPUT_LOG), overwrite=True)

fig_log = reject_log.plot(orientation="horizontal", show=False)
fig_log.savefig(str(OUTPUT_LOG.with_suffix(".png")), dpi=150, bbox_inches="tight")
plt.close(fig_log)
```

<br>

## 6. Final Epoch Inspection

**Run this in a Jupyter Notebook after Step 5.**
Opens the MNE interactive epoch browser. Use `Page Down` / `Page Up` to scroll through all 280 channels (50 at a time).

```python
# %% Jupyter Epoch Inspection Block
import mne
from pathlib import Path

%matplotlib widget

# ============================================================================
# USER CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path("/path/to/your/project")
CLEAN_EPO = PROJECT_ROOT / "derivatives" / "sub-01" / "eeg_epochs" / "sub-01_epo_clean.fif"
# ============================================================================

epochs_clean = mne.read_epochs(str(CLEAN_EPO), preload=True, verbose="WARNING")

fig_browser = epochs_clean.plot(
    n_epochs=1,
    n_channels=50,
    events=False,
    title="Interactive Epoch Inspector (Post-AutoReject)",
)
```


