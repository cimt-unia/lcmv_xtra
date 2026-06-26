# lcmv_xtra/__init__.py

from ._config import *  

# Main workflow functions 
from .source_estimation import execute_source_estimation
from .utils import download_fsaverage  
from .atlas_extraction import gt_extraction, difumo_extraction
from .cimt_atlas import cimt_extraction

# Tensor / ML Aggregation
from .tensor import (
    scan_eeg_paths,      
    save_study_tensor,
    assemble_tensor       
)

from .connectivity import (
    # GT Connectivity
    compute_gt_motor_connectivity,
    compute_gt_full_connectivity,
    get_motor_roi_metadata,
    
    # DiFuMo Connectivity
    compute_difumo_connectivity,
    
    # CIMT Connectivity (NEW)
    compute_cimt_full_connectivity,
    compute_cimt_motor_connectivity,
    get_cimt_motor_network_metadata
)

from .viz import (  
    visualize_source_at_coordinate,
    plot_mni_orthoview,
    compute_psd
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
    
    # Tensor / ML Aggregation
    "scan_eeg_paths",      
    "save_study_tensor",
    "assemble_tensor",     
    
    # Connectivity Analysis (GT, DiFuMo, CIMT)
    "compute_gt_motor_connectivity",
    "compute_gt_full_connectivity", 
    "get_motor_roi_metadata",
    "compute_difumo_connectivity",
    "compute_cimt_full_connectivity",      
    "compute_cimt_motor_connectivity",     
    "get_cimt_motor_network_metadata",     
    
    # Visualization & Utils
    "visualize_source_at_coordinate",  
    "plot_mni_orthoview",
    "compute_psd"
]
