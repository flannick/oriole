from __future__ import annotations

from dataclasses import dataclass

from .skip_stats import SkipStats


@dataclass
class TridentStats:
    stats: list[SkipStats]

    @classmethod
    def new(cls, x0: float, x1: float) -> "TridentStats":
        skip_stats = SkipStats.new(x0, x1)
        return cls(stats=[skip_stats, SkipStats.new(x0, x1), SkipStats.new(x0, x1)])

    def add(self, value: float) -> None:
        self.stats[2].add(value)
        if self.stats[2].n >= 2 * self.stats[1].n:
            mean = self.mean()
            std_dev = self.std_dev()
            stats_new = SkipStats.new(mean - std_dev, mean + std_dev)
            stats_old_2 = self.stats[2]
            self.stats[2] = stats_new
            stats_old_1 = self.stats[1]
            self.stats[1] = stats_old_2
            self.stats[0] = stats_old_1

    def mean(self) -> float:
        total_sum = self.stats[0].sum + self.stats[1].sum + self.stats[2].sum
        total_n = self.stats[0].n + self.stats[1].n + self.stats[2].n
        return total_sum / total_n

    def variance(self) -> float:
        total_var = (
            self.stats[0].var_sum + self.stats[1].var_sum + self.stats[2].var_sum
        )
        total_n = self.stats[0].n + self.stats[1].n + self.stats[2].n
        return total_var / total_n

    def std_dev(self) -> float:
        return self.variance() ** 0.5

