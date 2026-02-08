# lcmv_xtra/__init__.py

from ._config import *  

# Main workflow functions 
from .source_estimation import execute_source_estimation
from .utils import download_fsaverage  
from .atlas_extraction import gt_extraction, difumo_extraction
from .connectivity import (
    compute_gt_motor_connectivity,
    compute_gt_full_connectivity,
    compute_difumo_connectivity,
    get_motor_roi_metadata
)
from .viz import (  
    visualize_source_at_coordinate,
    plot_mni_orthoview,
    compute_psd
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
    "get_motor_roi_metadata",
    "visualize_source_at_coordinate",  
    "plot_mni_orthoview",
    "compute_psd"
]
