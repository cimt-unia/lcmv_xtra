# lcmv_xtra/__init__.py

from ._config import *  

# Main workflow functions 
from .source_estimation import execute_source_estimation
from .utils import download_fsaverage  
from .atlas_extraction import gt_extraction, difumo_extraction
from .cimt_atlas import cimt_extraction

# Tensor / ML Aggregation (Standard - Atlas based)
from .tensor import (
    scan_eeg_paths,      
    save_study_tensor,
    assemble_tensor       
)

# Tensor / ML Aggregation (Custom - MNI coordinate based)
from .custom_tensor import (
    assemble_custom_tensor,
    extract_custom_roi_time_courses,
)

from .connectivity import (
    compute_gt_motor_connectivity,
    compute_gt_full_connectivity,
    get_motor_roi_metadata,
    compute_difumo_connectivity,
    compute_cimt_full_connectivity,
    compute_cimt_motor_connectivity,
    get_cimt_motor_network_metadata
)

from .viz import (
    plot_mni_orthoview,
    plot_cimt_rois,
    plot_group_psd_comparison,
)

__version__ = "0.2.0"

__all__ = [
    # Core Pipeline
    "execute_source_estimation",
    "download_fsaverage", 
    
    # Atlas Extractions
    "gt_extraction",
    "difumo_extraction",
    "cimt_extraction",  
    
    # Tensor / ML Aggregation (Standard)
    "scan_eeg_paths",      
    "save_study_tensor",
    "assemble_tensor",     
    
    # Tensor / ML Aggregation (Custom MNI Coordinates)
    "assemble_custom_tensor",
    "extract_custom_roi_time_courses",
    
    # Connectivity Analysis
    "compute_gt_motor_connectivity",
    "compute_gt_full_connectivity", 
    "get_motor_roi_metadata",
    "compute_difumo_connectivity",
    "compute_cimt_full_connectivity",      
    "compute_cimt_motor_connectivity",     
    "get_cimt_motor_network_metadata",     
    
    # Visualization
    "plot_mni_orthoview",
    "plot_cimt_rois",
    "plot_group_psd_comparison",
]
