# lcmv_xtra/__init__.py

from ._config import *  # Set env vars first

# Main workflow functions (public API)
from .source_estimation import execute_source_estimation
from .utils import download_fsaverage  # ← Keep this in utils where it belongs
from .atlas_extraction import gt_extraction, difumo_extraction
from .connectivity import (
    compute_gt_motor_connectivity,
    compute_gt_full_connectivity,
    compute_difumo_connectivity,
    get_motor_roi_metadata
)

__version__ = "0.1.0"
__all__ = [
    "execute_source_estimation",
    "download_fsaverage", 
    "gt_extraction",
    "difumo_extraction",
    "compute_gt_motor_connectivity",
    "compute_gt_full_connectivity", 
    "compute_difumo_connectivity",
    "get_motor_roi_metadata"
]

