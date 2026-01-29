from __future__ import annotations

from datetime import timedelta


def format_duration(duration: timedelta) -> str:
    total_nanos = int(duration.total_seconds() * 1_000_000_000)
    if total_nanos == 0:
        return "0s"
    if total_nanos < 1_000:
        return f"{total_nanos}ns"
    total_micros = total_nanos // 1_000
    if total_micros < 1_000:
        return f"{total_micros}.{total_nanos % 1_000:0>3}us"
    total_millis = total_micros // 1_000
    if total_millis < 1_000:
        return f"{total_millis}.{total_nanos % 1_000:0>3}ms"
    total_secs = total_millis // 1_000
    if total_secs < 60:
        return f"{total_secs}.{total_millis % 1_000:0>3}s"
    mins = total_secs // 60
    if mins < 60:
        return f"{mins}m{total_secs % 60}s"
    hours = mins // 60
    if hours < 24:
        return f"{hours}h{mins % 60}m{total_secs % 60}s"
    days = hours // 24
    if days < 7:
        return f"{days}d{hours % 24}h{mins % 60}m"
    weeks = days // 7
    return f"{weeks}w{days % 7}d{hours % 24}h"
