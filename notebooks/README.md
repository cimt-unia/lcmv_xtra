# EEG Preprocessing Framework for LCMV Source Reconstruction

This guide explains how to prepare BEL 280-channel EEG data for the lcmv_xtra source reconstruction library using xeegkit. 

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
"""Standardize and conservatively clean continuous EEG."""
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

    raw = mne.io.read_raw_edf(str(RAW_EEG_FILE), preload=True, verbose="WARNING") # EDF TYPE

    # raw = mne.io.read_raw_egi(str(RAW_EEG_FILE), preload=True, verbose="WARNING") # MFF TYPE

    # raw = mne.io.read_raw_fifstr(str(RAW_EEG_FILE), preload=True, verbose="WARNING") # FIF TYPE

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



<br>

## 2. User Trimming

Use any method or tool you prefer to segment, align, or select your pre-cleaned data. There are no restrictions on how you do this.

Critical requirement: If you create epochs, you must concatenate them into a single continuous Raw object before proceeding. The final cleaning pipeline requires a gap-free continuous signal to calibrate ASR and compute spatial filters correctly.

Save the resulting continuous .fif file and set its path as TRIMMED_FIF.

<br>

## 3.Final Cleaning

Accepts **any** pre-cleaned FIF produced by the user's trimming step. Applies ASR, STAR, SNS, and a refined ICLabel pass. Edit only the `USER CONFIGURATION` section at the top.

```python
"""MEEGKit + ICLabel on user-trimmed data."""
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

# Region-specific line noise frequency
# US/Canada/Japan (60Hz regions): 60.0
# Europe/UK/Australia/Most of Asia: 50.0

NOTCH_FREQ: float = 50.0  

# MEEGKit parameters (ASR + STAR + SNS)
MEEGKIT_PARAMS = {
    "highpass_filter": 1.0,
    "low_pass_filter": 100.0,
    "notch_filter_freq": NOTCH_FREQ,  
    "mad_threshold": 20.0,
    "min_amplitude_uv": 0.5,
    "asr_cutoff": 3.5,                
    "star_thresh": 2.5,
    "sns_neighbors": 8,
    "drop_cz": False,                  
    "interpolate_bads": True,
    "generate_report": True,
}

# ICLabel refinement parameters
ICALABEL_PARAMS = {
    "mad_threshold": 50.0,            
    "min_amplitude_uv": 0.5,
    "n_components": 0.99,
    "random_state": 42,
    "interpolate_bads": True,
    "generate_report": True,
}

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



<br>

## 4. Data Inspection

**Run this after both cleaning steps are complete.** This single script loads both the pre-cleaned and final cleaned FIF files side-by-side for comparative visual QA. This is mandatory to verify that cleaning preserved neural signals and did not introduce artifacts before proceeding to LCMV source reconstruction. Edit only the `USER CONFIGURATION` section at the top.

```python
""" Comparative visual QA of pre-clean and final clean."""
from __future__ import annotations

import warnings
from pathlib import Path

import mne

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# USER CONFIGURATION — EDIT THIS SECTION ONLY
# ============================================================================

PRECLean_FIF = Path("/derivatives/sub-01/sub-01_preclean_raw.fif")
CLEANED_FIF = Path("/derivatives/sub-01/sub-01_trimmed_raw_eeg.fif")

# ============================================================================



def _print_summary(label: str, raw: mne.io.Raw) -> None:
    """Print standardized summary for a Raw object."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Channels: {len(raw.ch_names)}")
    print(f"  Duration: {raw.n_times / raw.info['sfreq']:.1f}s")
    print(f"  Highpass: {raw.info.get('highpass', 'N/A')} Hz")
    print(f"  Lowpass:  {raw.info.get('lowpass', 'N/A')} Hz")
    print(f"  Bads:     {raw.info['bads']}")
    print(f"{'=' * 60}")



"""Load and plot both pre-cleaned and final cleaned data for comparative QA."""
if not PRECLEAN_FIF.exists():
    raise FileNotFoundError(f"Pre-cleaned FIF not found: {PRECLean_FIF}")
if not CLEANED_FIF.exists():
    raise FileNotFoundError(f"Final cleaned FIF not found: {CLEANED_FIF}")

raw_pre = mne.io.read_raw_fif(str(PRECLEAN_FIF), preload=True, verbose="WARNING")
raw_clean = mne.io.read_raw_fif(str(CLEANED_FIF), preload=True, verbose="WARNING")

_print_summary("PRE-CLEAN (MAD + ICA)", raw_pre)
_print_summary("FINAL CLEAN (ASR+STAR+SNS+ICLabel)", raw_clean)

# Open both plots for side-by-side comparison
raw_pre.plot(n_channels=45, title="PRE-CLEAN QA", verbose="WARNING")
raw_clean.plot(n_channels=45, title="FINAL CLEAN QA", verbose="WARNING")


```



<br>
