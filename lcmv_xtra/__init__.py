# lcmv_xtra/__init__.py

from ._config import *  # Set env vars first

# Main workflow functions (public API)
from .utils import download_fsaverage 
from .source_estimation import execute_source_estimation
from .atlas_extraction import difumo_extraction, gt_extraction

__version__ = "0.1.0"
__all__ = [
    "execute_source_estimation",
    "download_fsaverage",
    "difumo_extraction", 
    "gt_extraction"
]
