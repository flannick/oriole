from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Stats:
    n: int = 0
    sum: float = 0.0
    var_sum: float = 0.0

    def add(self, value: float) -> None:
        if self.n == 0:
            self.n += 1
            self.sum += value
            return
        mean_previous = self.sum / self.n
        self.n += 1
        self.sum += value
        mean = self.sum / self.n
        self.var_sum += (value - mean_previous) * (value - mean)

    def mean(self) -> float | None:
        if self.n == 0:
            return None
        return self.sum / self.n

    def variance(self) -> float | None:
        if self.n < 2:
            return None
        return self.var_sum / self.n

