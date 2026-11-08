"""mathbridge: analysis, Fourier, PDEs, and geometry over NumPy/SciPy/SymPy."""
from . import fourier, functional, geometry, pde
from .result import Result

__version__ = "0.1.0"
__all__ = ["functional", "fourier", "pde", "geometry", "Result"]
