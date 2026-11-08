"""Common result container: a value, the solution steps, and an optional plot."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class Result:
    """Returned by mathbridge functions.

    value  -- the numerical / symbolic answer
    steps  -- human-readable solution steps (always recorded)
    """

    value: Any
    steps: List[str] = field(default_factory=list)
    _plotter: Optional[Callable[[], Any]] = field(default=None, repr=False)

    def explain(self) -> str:
        return "\n".join(f"{i}. {s}" for i, s in enumerate(self.steps, 1))

    def plot(self):
        if self._plotter is None:
            raise ValueError("No plot is available for this result.")
        return self._plotter()

    def __float__(self):
        return float(self.value)

    def __repr__(self):
        return f"Result(value={self.value!r}, steps={len(self.steps)})"


def finish(result: Result, explain: bool, plot: bool):
    """Apply the explain=/plot= flags shared by every public function."""
    if explain:
        print(result.explain())
    if plot:
        result.plot()
    return result
