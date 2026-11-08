"""Heat and wave equations with homogeneous Dirichlet boundary conditions."""
from .heat import heat_fd, heat_separation
from .wave import wave_fd, wave_separation

__all__ = ["heat_separation", "heat_fd", "wave_separation", "wave_fd"]
