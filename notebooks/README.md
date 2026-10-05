# EEG Preprocessing Framework for LCMV Source Reconstruction

This guide explains how to prepare BEL 280-channel EEG data for the lcmv_xtra source reconstruction library using xeegkit. The workflow is split into two simple stages with your own processing in between. Each script is standalone and includes its own settings at the top, so no extra configuration files are needed.

<br>

### Introduction

```text
Raw EEG ──► [Step 1: Pre-Clean] ──► [Step 2: Trim/Epoch (User)] ──► [Step 3: Final Clean] ──► Analysis-Ready FIF
```

| Step | Input | Operation | Output |
| :--- | :--- | :--- | :--- |
| **1. Pre-Clean** | Raw continuous EEG | Standardize + Conservative ICA | `*_preclean_raw.fif` |
| **2. Trim/Epoch** | Pre-cleaned FIF | User-defined segmentation | `*_trimmed_raw.fif` |
| **3. Final Clean** | Trimmed/concatenated FIF | ASR + STAR + SNS + ICLabel | `*_eeg.fif` |

<br>

## 1. Raw Data Pre-Cleaning

Runs on the **full continuous recording** before any trimming. This maximizes data available for ICA decomposition and avoids filter edge artifacts at future epoch boundaries. Edit only the `USER CONFIGURATION` section at the top.

```python
"""script1_preclean.py: Standardize and conservatively clean continuous EEG."""
from __future__ import annotations

import logging
from pathlib import Path

import mne
from xeeg_kit import BELStandardizer
from xeeg_kit.bel_pipeline import DEFAULT_RENAME_MAP
from xeeg_kit.pre_cleaning import auto_preclean

# ============================================================================
# USER CONFIGURATION — EDIT THIS SECTION ONLY
# ============================================================================

# Paths
RAW_EEG_FILE = Path("/data/raw/sub-01_task.edf")
GPSC_FILE = Path("/data/montage/ghw280_from_egig.gpsc")
OUTPUT_DIR = Path("/derivatives/sub-01")
OUTPUT_NAME = "sub-01_preclean_raw"

# Region-specific line noise frequency
# US/Canada/Japan (60Hz regions): 60.0
# Europe/UK/Australia/Most of Asia: 50.0
NOTCH_FREQ: float = 50.0  # ← SET TO MATCH YOUR ACQUISITION SITE

# Conservative pre-clean parameters
PRECLEAN_PARAMS = {
    "mad_threshold": 20.0,        # High → fewer channels flagged
    "artifact_threshold": 0.80,   # High → fewer ICA components rejected
    "n_components": 0.99,         # Variance explained for ICA
    "highpass": 1.0,              # High-pass filter (Hz)
    "lowpass": 100.0,             # Low-pass filter (Hz)
    "notch_freq": NOTCH_FREQ,     # Line noise removal
}

# ============================================================================
# END USER CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_preclean() -> None:
    """Load, standardize, and pre-clean full continuous EEG."""
    logger.info("=" * 70)
    logger.info("Step 1: Pre-Clean Continuous EEG")
    logger.info("=" * 70)

    if not RAW_EEG_FILE.exists():
        raise FileNotFoundError(f"Raw EEG not found: {RAW_EEG_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_edf(str(RAW_EEG_FILE), preload=True, verbose="WARNING")
    logger.info("Loaded: %.1f Hz, %d channels", raw.info["sfreq"], len(raw.ch_names))

    standardizer = BELStandardizer(gpsc_file=GPSC_FILE, rename_map=DEFAULT_RENAME_MAP)
    raw = standardizer.standardize(raw)

    logger.info("Running auto_preclean on full continuous EEG...")
    _ = auto_preclean(
        raw, OUTPUT_DIR, OUTPUT_NAME,
        drop_channels=["Cz"],  # Remove BEL 280 hardware ref before avg reference
        **PRECLEAN_PARAMS,
    )

    output_path = OUTPUT_DIR / f"{OUTPUT_NAME}.fif"
    logger.info("Pre-clean complete. Inspect output before trimming.")
    logger.info("Output: %s", output_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    run_preclean()
```

### Fixed Parameter Rationale

-   **`notch_freq`**: Must match the local grid (50 Hz EU / 60 Hz US). Using the wrong frequency fails to suppress line noise and introduces artifacts.
-   **`mad_threshold=20.0`**: Prevents flagging channels with legitimate task-related variance (e.g., motor cortex during movement).
-   **`artifact_threshold=0.80`**: Retains mixed neural/artifact ICA components. Aggressive rejection is deferred to Step 3, where ASR provides spatially informed cleanup.
-   **`drop_channels=["Cz"]`**: Prevents rank deficiency during average referencing. The benign warning (`Requested drop channels not found`) occurs when the standardizer has already renamed `Cz`; this is safe to ignore.

<br>

## 2. User Trimming/Epoch

Between Step 1 and Step 3, users perform their own epoching, TTL alignment, concatenation, or segment selection. **No `xeegkit` code is prescribed here.** The only requirement is that the output is a valid MNE-compatible FIF file whose path is set as `TRIMMED_FIF` in Script 3.

## 3.Final Cleaning

Accepts **any** pre-cleaned FIF produced by the user's trimming step. Applies ASR, STAR, SNS, and a refined ICLabel pass. Edit only the `USER CONFIGURATION` section at the top.

```python
"""script3_final_clean.py: MEEGKit + ICLabel on user-trimmed data."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from xeeg_kit import preprocess_bel_trials

# ============================================================================
# USER CONFIGURATION — EDIT THIS SECTION ONLY
# ============================================================================

# Paths
TRIMMED_FIF = Path("/derivatives/sub-01/sub-01_trimmed_raw.fif")  # Your trimmed input
GPSC_FILE = Path("/data/montage/ghw280_from_egig.gpsc")
OUTPUT_DIR = Path("/derivatives/sub-01")

# Region-specific line noise frequency (MUST MATCH STEP 1)
# US/Canada/Japan (60Hz regions): 60.0
# Europe/UK/Australia/Most of Asia: 50.0
NOTCH_FREQ: float = 50.0  # ← MUST MATCH STEP 1 VALUE

# MEEGKit parameters (ASR + STAR + SNS)
MEEGKIT_PARAMS = {
    "highpass_filter": 1.0,
    "low_pass_filter": 100.0,
    "notch_filter_freq": NOTCH_FREQ,  # Must match Step 1
    "mad_threshold": 20.0,
    "min_amplitude_uv": 0.5,
    "asr_cutoff": 3.5,                # Conservative for task/movement data
    "star_thresh": 2.5,
    "sns_neighbors": 8,
    "drop_cz": False,                  # Cz already removed in Step 1
    "interpolate_bads": True,
    "generate_report": True,
}

# ICLabel refinement parameters
ICALABEL_PARAMS = {
    "mad_threshold": 50.0,             # Higher than Step 1 (refinement pass)
    "min_amplitude_uv": 0.5,
    "n_components": 0.99,
    "random_state": 42,
    "interpolate_bads": True,
    "generate_report": True,
}

# ============================================================================
# END USER CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def run_final_clean() -> Dict[str, Path]:
    """Apply final MEEGKit + ICLabel pipeline to trimmed FIF."""
    logger.info("=== Step 3: Final BEL Pipeline Cleaning ===")

    if not TRIMMED_FIF.exists():
        raise FileNotFoundError(
            f"Trimmed FIF not found: {TRIMMED_FIF}\n"
            "Complete your trimming/epoching step first."
        )

    result = preprocess_bel_trials(
        data_dir=OUTPUT_DIR,
        output_dir=OUTPUT_DIR,
        gpsc_path=GPSC_FILE,
        meegkit_params=MEEGKIT_PARAMS,
        icalabel_params=ICALABEL_PARAMS,
        pattern=TRIMMED_FIF.name,
        overwrite=True,
        verbose=True,
    )

    logger.info("Final cleaning complete: %d file(s) processed", len(result))
    return result


if __name__ == "__main__":
    results = run_final_clean()
    for name, path in results.items():
        logger.info("Output: %s → %s", name, path)
```

### Key Design Decisions

-   **Input Agnostic**: Accepts any FIF regardless of whether it contains continuous, epoched, or concatenated data.
-   **ASR Calibration**: Automatically selects the cleanest segment from the trimmed data. Always verify the timestamp in logs—task-active segments can contaminate the baseline.
-   **Refined ICLabel**: `mad_threshold=50.0` catches residuals missed by Step 1’s conservative threshold without over-correcting.
-   **No Redundant Cz Removal**: `drop_cz=False` because hardware reference removal is handled exclusively in Step 1.

