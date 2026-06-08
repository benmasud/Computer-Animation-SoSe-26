"""Motion player package for parsing and visualizing motion capture data."""

from .amc_parser import *
from .Viewer3D import Viewer

__version__ = "1.0.0"
__all__ = ['parse_asf', 'parse_amc', 'Viewer']