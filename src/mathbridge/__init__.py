"""mathbridge: one friendly layer over NumPy/SciPy for analysis and Fourier."""
from . import fourier, functional
from .result import Result

__version__ = "0.1.0"
__all__ = ["functional", "fourier", "Result"]
