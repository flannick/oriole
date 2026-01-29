from __future__ import annotations

from datetime import datetime

from .train.param_meta_stats import Summary
from .util.duration_format import format_duration


class Reporter:
    def __init__(self) -> None:
        self.start_time = datetime.now()
        self.start_time_round = datetime.now()
        self.time_since_last_report = datetime.now()

    def reset_round_timer(self) -> None:
        self.start_time_round = datetime.now()

    def report(self, summary: Summary, i_cycle: int, i_iteration: int, n_samples: int) -> None:
        duration_round = datetime.now() - self.start_time_round
        secs_elapsed = duration_round.total_seconds() or 1e-9
        elapsed_round = format_duration(duration_round)
        duration_total = datetime.now() - self.start_time
        elapsed_total = format_duration(duration_total)
        steps_per_sec = (i_iteration + 1) * n_samples / secs_elapsed
        print(
            "Cycle {}, iteration {}: collected {} samples in {}, "
            "which is {} iterations per second and thread. Total time is {}".format(
                i_cycle, i_iteration, n_samples, elapsed_round, steps_per_sec, elapsed_total
            )
        )
        print(summary)
        self.time_since_last_report = datetime.now()

