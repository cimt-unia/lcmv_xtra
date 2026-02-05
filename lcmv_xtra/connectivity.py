# lcmv_xtra/connectivity.py

import lcmv_xtra  
import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import datasets
from mne_connectivity import spectral_connectivity_epochs


# UTILS

def get_frequency_bands():
    """Get standard frequency band definitions with lowercase keys for easier typing."""
    return {
        "theta": (4, 8), 
        "alpha": (8, 12), 
        "low_beta": (13, 20),
        "high_beta": (20, 30), 
        "beta": (13, 30), 
        "low_gamma": (30, 60),
        "high_gamma": (60, 120), 
        "gamma": (30, 100)
    }

def compute_connectivity_matrix(
    data: np.ndarray,
    fmin: float,
    fmax: float,
    sfreq: float = 500.0,
    method: str = 'wpli2_debiased'
) -> np.ndarray:
    """Compute connectivity matrix using MNE."""
    con = spectral_connectivity_epochs(
        data=data,
        method=method,
        mode='multitaper',
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose=False
    )
    matrix = con.get_data(output='dense').squeeze()
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    return matrix


# GLASSER & TIAN ATLAS

def _get_bundled_gt_roi_file() -> Path:
    """Get path to bundled GT atlas ROI file."""
    package_dir = Path(lcmv_xtra.__file__).parent  
    roi_file = package_dir / 'data' / 'gt_atlas' / 'roi_labels.csv'
    if not roi_file.exists():
        raise FileNotFoundError(f"Bundled GT ROI file not found: {roi_file}")
    return roi_file


def compute_gt_full_connectivity(
    epochs_data: np.ndarray,
    band_name: str = "beta",
    sfreq: float = 500.0,
    method: str = 'wpli2_debiased'
) -> pd.DataFrame:
    roi_file = _get_bundled_gt_roi_file()
    roi_df = pd.read_csv(roi_file)
    roi_names = roi_df['roi_name'].tolist()
    
    if epochs_data.shape[1] != len(roi_names):
        raise ValueError(
            f"ROI count mismatch: expected {len(roi_names)}, "
            f"got {epochs_data.shape[1]}"
        )
    
    # Get frequency bands
    bands = get_frequency_bands()
    
    if band_name in bands:
        fmin, fmax = bands[band_name]
    else:
        available_bands = list(bands.keys())
        raise ValueError(f"Unknown band: '{band_name}'. Available options: {available_bands}")
    
    matrix = compute_connectivity_matrix(epochs_data, fmin, fmax, sfreq, method)
    return pd.DataFrame(matrix, index=roi_names, columns=roi_names)


def select_motor_rois() -> dict:
    """
    Select ROIs using functional system metadata from bundled CSV file.
    No need for users to provide roi_file parameter.
    """
    roi_file = _get_bundled_gt_roi_file()
    roi_df = pd.read_csv(roi_file)
    FULL_ROI_NAMES = roi_df['roi_name'].tolist()
    
    motor_rois = roi_df[
        (roi_df['functional_system'] == 'Motor') |
        (roi_df['sub_system'].isin(['Primary', 'Premotor', 'Supplementary', 'Eye']))
    ]['roi_name'].tolist()

    basal_rois = roi_df[
        (roi_df['functional_system'] == 'BasalGanglia') |
        (roi_df['region_full_name'].str.contains(
            r'Putamen|Caudate|Globus Pallidus|Nucleus Accumbens', 
            case=False, na=False
        ))
    ]['roi_name'].tolist()

    executive_rois = roi_df[
        (roi_df['functional_system'] == 'Frontoparietal') &
        (roi_df['sub_system'].isin(['DLPFC', 'IFJ', 'IFS', 'VLPFC']))
    ]['roi_name'].tolist()

    TARGET_ROIS = sorted(set(motor_rois + basal_rois + executive_rois))
    TARGET_INDICES = [FULL_ROI_NAMES.index(r) for r in TARGET_ROIS]
    
    return {
        'target_rois': TARGET_ROIS,
        'target_indices': TARGET_INDICES,
        'full_roi_names': FULL_ROI_NAMES
    }

def get_motor_roi_metadata(save_to: Path | str | None = None) -> pd.DataFrame:
    """
    Return a DataFrame of GT atlas ROIs selected for the motor-basal-executive network,
    with a 'new_index' column reflecting their position in the reduced connectivity matrix.
    
    Parameters
    ----------
    save_to : Path or str, optional
        If provided, save the result as a CSV at this path.
    
    Returns
    -------
    pd.DataFrame
        Columns: ['new_index', 'roi_name', 'hemisphere', 'region_full_name',
                  'functional_system', 'sub_system']
    """
    roi_file = _get_bundled_gt_roi_file()
    roi_df = pd.read_csv(roi_file)

    # Apply same selection logic as in select_motor_rois()
    motor_mask = (
        (roi_df['functional_system'] == 'Motor') |
        (roi_df['sub_system'].isin(['Primary', 'Premotor', 'Supplementary', 'Eye']))
    )
    basal_mask = (
        (roi_df['functional_system'] == 'BasalGanglia') |
        (roi_df['region_full_name'].str.contains(
            r'Putamen|Caudate|Globus Pallidus|Nucleus Accumbens', 
            case=False, na=False
        ))
    )
    executive_mask = (
        (roi_df['functional_system'] == 'Frontoparietal') &
        (roi_df['sub_system'].isin(['DLPFC', 'IFJ', 'IFS', 'VLPFC']))
    )

    selected_df = roi_df[motor_mask | basal_mask | executive_mask].copy()
    selected_df = selected_df.sort_values('roi_name').reset_index(drop=True)
    selected_df.insert(0, 'new_index', range(len(selected_df)))

    if save_to is not None:
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)
        selected_df.to_csv(save_to, index=False)

    return selected_df


def compute_gt_motor_connectivity(
    epochs_data: np.ndarray,
    band_name: str = "beta",
    sfreq: float = 500.0,
    method: str = 'wpli2_debiased'
) -> pd.DataFrame:
    roi_info = select_motor_rois()
    target_indices = roi_info['target_indices']
    roi_names = roi_info['target_rois']
    
    if epochs_data.shape[1] != len(roi_info['full_roi_names']):
        raise ValueError(
            f"ROI count mismatch: expected {len(roi_info['full_roi_names'])}, "
            f"got {epochs_data.shape[1]}"
        )
    
    selected_data = epochs_data[:, target_indices, :]
    
    # Get frequency bands
    bands = get_frequency_bands()
    
    if band_name in bands:
        fmin, fmax = bands[band_name]
    else:
        available_bands = list(bands.keys())
        raise ValueError(f"Unknown band: '{band_name}'. Available options: {available_bands}")
    
    matrix = compute_connectivity_matrix(selected_data, fmin, fmax, sfreq, method)
    return pd.DataFrame(matrix, index=roi_names, columns=roi_names)


# DIFUMO ATLAS

def get_difumo_names(dimension: int = 512, resolution_mm: int = 2) -> list[str]:
    try:
        atlas = datasets.fetch_atlas_difumo(dimension=dimension, resolution_mm=resolution_mm)
        return atlas.labels['difumo_names'].astype(str).tolist()
    except Exception:
        return [f"Component_{i}" for i in range(dimension)]


def motor_rois() -> list[int]:
    Motor_M1 = [40, 86, 198, 268, 305, 437, 458, 465]
    Motor_SMA_Premotor = [17, 18, 288, 291, 296, 297, 302, 305, 314, 315, 335, 375, 379, 448]
    Motor_Medial = [101, 102, 388, 409, 498]
    Thalamus = [70, 73, 297, 334, 414, 420] 
    Basal_Ganglia = [30, 53, 224, 260, 405, 422, 109, 110, 315, 331, 467, 479, 55, 71, 307, 223]  
    Cerebellum_Motor = [43, 47, 83, 84, 127, 183, 220, 221, 295, 304, 310, 311, 374, 378, 381, 403, 441, 490, 491]
    Somatosensory = [44, 131, 210, 411, 413, 436]
    Executive_Control = [3, 85, 104, 148, 184, 337, 377, 446, 447, 506, 507]
    Interoception = [2, 387, 358, 389, 165, 469]
    Error_Monitoring = [185, 219, 326, 473, 492]
    
    return sorted(set(
        Motor_M1 + Motor_SMA_Premotor + Motor_Medial + Thalamus +
        Basal_Ganglia + Cerebellum_Motor + Somatosensory +
        Executive_Control + Interoception + Error_Monitoring
    ))


def compute_difumo_connectivity(
    epochs_data: np.ndarray,
    band_name: str = "beta",
    dimension: int = 512,
    resolution_mm: int = 2,
    sfreq: float = 500.0,
    method: str = 'wpli2_debiased'
) -> pd.DataFrame:
    all_names = get_difumo_names(dimension, resolution_mm)
    selected_indices = motor_rois()
    roi_names = [all_names[i] for i in selected_indices]
    
    if epochs_data.shape[1] != dimension:
        raise ValueError(f"Expected {dimension} DiFuMo components, got {epochs_data.shape[1]}")
    
    selected_data = epochs_data[:, selected_indices, :]
    bands = get_frequency_bands()
    if band_name not in bands:
        available_bands = list(bands.keys())
        raise ValueError(f"Unknown band: '{band_name}'. Available options: {available_bands}")
    fmin, fmax = bands[band_name]
    
    matrix = compute_connectivity_matrix(selected_data, fmin, fmax, sfreq, method)
    return pd.DataFrame(matrix, index=roi_names, columns=roi_names)

