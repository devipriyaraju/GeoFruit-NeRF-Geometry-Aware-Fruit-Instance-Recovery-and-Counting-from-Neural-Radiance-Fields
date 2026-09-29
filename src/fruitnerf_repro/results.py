from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CountResult:
    tree: str
    gt: int
    paper: int
    ours: int

    @property
    def paper_error(self) -> int:
        return self.paper - self.gt

    @property
    def ours_error(self) -> int:
        return self.ours - self.gt

    @property
    def paper_ratio(self) -> float:
        return self.paper / self.gt

    @property
    def ours_ratio(self) -> float:
        return self.ours / self.gt


MAIN_RESULTS = [
    CountResult("tree_01", 179, 147, 168),
    CountResult("tree_02", 113, 86, 98),
]
