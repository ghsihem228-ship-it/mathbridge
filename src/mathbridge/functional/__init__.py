from .convergence import convergence_report
from .operators import galerkin_matrix, integral_operator, operator_norm, spectrum
from .spaces import distance, gram_schmidt, inner_product, norm, project

__all__ = [
    "inner_product", "norm", "distance", "gram_schmidt", "project",
    "galerkin_matrix", "integral_operator", "spectrum", "operator_norm",
    "convergence_report",
]
