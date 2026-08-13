# lcmv_xtra/__init__.py

from ._config import *  

# Main workflow functions 
from .source_estimation import execute_source_estimation
from .utils import download_fsaverage  
from .atlas_extraction import gt_extraction, difumo_extraction
from .cimt_atlas import cimt_extraction

# Tensor / ML Aggregation (Standard - Atlas based)
from .tensor import (
    make_subject_list,
    scan_eeg_paths,      
    save_study_tensor,
    assemble_tensor       
)

# Tensor / ML Aggregation (Custom - MNI coordinate based)
from .custom_tensor import (
    assemble_custom_tensor,
    extract_custom_roi_time_courses,
)

# Atlas-Constrained Tensor Assembly (CIMT Before Inverse)
from .atlas_tensor import assemble_atlas_tensor

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

from .coreg import plot_coregistration

from .epoch_tensor import execute_epoch_tensor

from .source_estimation_atlas import (
    execute_source_estimation_atlas,
    lcmv_beamformer_cimt,
    reduce_leadfield_to_cimt,
    lookup_mni_coordinate,
    lookup_multiple_coordinates,
)

__version__ = "0.1.0"

__all__ = [
    # Core Pipeline
    "execute_source_estimation",
    "download_fsaverage", 
    
    # Atlas Extractions
    "gt_extraction",
    "difumo_extraction",
    "cimt_extraction",  
    
    # Tensor / ML Aggregation (CIMT Atlas Standard)
    "make_subject_list",
    "scan_eeg_paths",      
    "save_study_tensor",
    "assemble_tensor",     
    
    # Tensor / ML Aggregation (Custom MNI Coordinates)
    "assemble_custom_tensor",
    "extract_custom_roi_time_courses",

    # Epoched Source Estimation (Custom MNI Coordinates)
    "execute_epoch_tensor",
    
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
    "plot_coregistration",

    # Atlas-Constrained Source Estimation (CIMT Before Inverse)
    "execute_source_estimation_atlas",
    "lcmv_beamformer_cimt",
    "reduce_leadfield_to_cimt",
    "lookup_mni_coordinate",
    "lookup_multiple_coordinates",

    # Tensor / ML Aggregation (CIMT Atlas-Before-Inverse)
    "assemble_atlas_tensor",

]
