from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SkipStats:
    n: int
    sum: float
    var_sum: float

    @classmethod
    def new(cls, x0: float, x1: float) -> "SkipStats":
        n = 2
        total = x0 + x1
        var_sum = 0.5 * (x1 - x0) ** 2
        return cls(n=n, sum=total, var_sum=var_sum)

    def add(self, value: float) -> None:
        mean_previous = self.sum / self.n
        self.n += 1
        self.sum += value
        mean = self.sum / self.n
        self.var_sum += (value - mean_previous) * (value - mean)

