# lcmv_xtra/__init__.py

from ._config import *  

# Main workflow functions 
from .source_estimation import execute_source_estimation
from .source_estimation_epochs import execute_source_estimation_epochs
from .utils import download_fsaverage  
from .atlas_extraction import gt_extraction, difumo_extraction
from .cimt_atlas import cimt_extraction

# Tensor / ML Aggregation (Standard - Atlas-based)
from .tensor import (
    make_subject_list,
    scan_eeg_paths,      
    save_study_tensor,
    assemble_tensor       
)

# Tensor / ML Aggregation (Epoch-based - 4D)
from .tensor_epochs import (
    assemble_tensor_epochs,
    save_study_tensor_epochs,
)

# Tensor / ML Aggregation (Custom - MNI coordinate-based)
from .custom_tensor import (
    assemble_custom_tensor,
    extract_custom_roi_time_courses,
)
from .custom_tensor_epochs import assemble_custom_tensor_epochs

# Native-Space Source Estimation & Tensor Assembly (Subject T1 BEM Forward Model)
from .source_estimation_epochs_native import execute_source_estimation_epochs_native
from .custom_tensor_epochs_native import assemble_custom_tensor_epochs_native

# FEM Native-Space Source Estimation & Tensor Assembly (SimNIBS Forward Model)
from .source_estimation_epochs_native_fem import execute_source_estimation_epochs_native_fem
from .custom_tensor_epochs_native_fem import assemble_custom_tensor_epochs_native_fem

# Atlas-Constrained Source Estimation & Tensor Assembly (CIMT Before Inverse)
from .source_estimation_atlas import (
    execute_source_estimation_atlas,
    lcmv_beamformer_cimt,
    reduce_leadfield_to_cimt,
    lookup_mni_coordinate,
    lookup_multiple_coordinates,
)
from .atlas_tensor import assemble_atlas_tensor

# Atlas-Constrained Source Estimation & Tensor Assembly (Epochs - Analytical LCMV)
from .source_estimation_epochs_atlas import (
    execute_source_estimation_epochs_atlas,
    lcmv_beamformer_epochs_atlas,
)
# FIX: Changed from .custom_tensor_epochs_atlas to match your actual GitHub filename
from .custom_tensor_atlas_epochs import (
    assemble_custom_atlas_tensor_epochs,
    map_custom_coords_to_atlas,
    save_custom_atlas_tensor_epochs,
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

from .coreg import plot_coregistration

from .tensor_variable import execute_variable_tensor

# Deep Learning Neural Beamformer (Optional: requires PyTorch)
try:
    from .dl_beamformer import (
        execute_source_estimation_atlas_pytorch,
        lcmv_beamformer_cimt_pytorch,
    )
    from .dl_tensor_epochs import assemble_dl_tensor_epochs
    from .source_estimation_epochs_atlas_dl import (
        execute_source_estimation_epochs_atlas_dl,
        lcmv_beamformer_epochs_atlas_dl,
    )
    # Matches your GitHub filename exactly
    from .custom_tensor_epochs_atlas_dl import (
        assemble_custom_atlas_tensor_epochs_dl,
        save_custom_atlas_tensor_epochs_dl,
    )
except ImportError:
    execute_source_estimation_atlas_pytorch = None
    lcmv_beamformer_cimt_pytorch = None
    assemble_dl_tensor_epochs = None
    execute_source_estimation_epochs_atlas_dl = None
    lcmv_beamformer_epochs_atlas_dl = None
    assemble_custom_atlas_tensor_epochs_dl = None
    save_custom_atlas_tensor_epochs_dl = None

__version__ = "0.1.0"

__all__ = [
    # Core Pipeline
    "execute_source_estimation",
    "execute_source_estimation_epochs",
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
    
    # Tensor / ML Aggregation (Epoch-based 4D)
    "assemble_tensor_epochs",
    "save_study_tensor_epochs",
    
    # Tensor / ML Aggregation (Custom MNI Coordinates)
    "assemble_custom_tensor",
    "extract_custom_roi_time_courses",
    "assemble_custom_tensor_epochs",

    # Native-Space Source Estimation & Tensor Assembly (Subject T1 BEM Forward Model)
    "execute_source_estimation_epochs_native",
    "assemble_custom_tensor_epochs_native",

    # FEM Native-Space Source Estimation & Tensor Assembly (SimNIBS Forward Model)
    "execute_source_estimation_epochs_native_fem",
    "assemble_custom_tensor_epochs_native_fem",

    # Trial Source Estimation
    "execute_variable_tensor",
    
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

    # Atlas-Constrained Source Estimation (Epochs - Analytical LCMV)
    "execute_source_estimation_epochs_atlas",
    "lcmv_beamformer_epochs_atlas",
    "assemble_custom_atlas_tensor_epochs",
    "map_custom_coords_to_atlas",
    "save_custom_atlas_tensor_epochs",

    # Deep Learning Neural Beamformer (requires PyTorch)
    "execute_source_estimation_atlas_pytorch",
    "lcmv_beamformer_cimt_pytorch",
    "assemble_dl_tensor_epochs",
    "execute_source_estimation_epochs_atlas_dl",
    "lcmv_beamformer_epochs_atlas_dl",
    "assemble_custom_atlas_tensor_epochs_dl",
    "save_custom_atlas_tensor_epochs_dl",
]
