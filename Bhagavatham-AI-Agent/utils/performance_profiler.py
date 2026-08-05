"""
Performance profiling utilities.

This module provides a lightweight profiler for measuring execution
time of different stages in the RAG pipeline.
"""

from __future__ import annotations


import time
from dataclasses import dataclass
from typing import Dict
from collections import OrderedDict



@dataclass
class PerformanceMetric:
    """Represents a measured execution stage."""

    name: str
    elapsed_seconds: float = 0.0
    item_count: int = 0


class PerformanceProfiler:
    """
    Simple performance profiler.

    Example
    -------
    profiler = PerformanceProfiler()

    profiler.start("Embedding")
    ...
    profiler.stop("Embedding")

    print(profiler.summary())
    """

    def __init__(self) -> None:
        self._start_times: Dict[str, float] = {}
        self._metrics: OrderedDict[str, PerformanceMetric] = OrderedDict()

    def start(self, stage: str) -> None:
        """
        Start timing a stage.

        Raises
        ------
        ValueError
        If the stage has already been started.
        """
        if stage in self._start_times:
            raise ValueError(
                f"Stage '{stage}' has already been started."
            )

        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str) -> None:
        """Stop timing a stage."""
        if stage not in self._start_times:
            raise ValueError(f"Stage '{stage}' was never started.")

        elapsed = time.perf_counter() - self._start_times.pop(stage)

        metric = self._metrics.setdefault(
            stage,
            PerformanceMetric(name=stage)
        )

        metric.elapsed_seconds = elapsed  

    def set_item_count(self, stage: str, count: int) -> None:
        """
        Record the number of items processed in a stage.

        Example:
        profiler.set_item_count("Embedding", 6889)
        """
        metric = self._metrics.setdefault(
            stage,
            PerformanceMetric(name=stage)
        )

        metric.item_count = count


    def get_elapsed(self, stage: str) -> float:
        """Return elapsed time for a stage."""
        if stage not in self._metrics:
            return 0.0

        return self._metrics[stage].elapsed_seconds

    

    def summary(self) -> str:
        """Return formatted timing summary."""

        total = self.total_time()

        lines = [
            "=" * 60,
            "Performance Summary",
            "=" * 60,
        ]

        for metric in self._metrics.values():

            lines.append(
                f"{metric.name:<25}"
                f"{metric.elapsed_seconds:>10.2f} sec"
            )

            if metric.item_count > 0:
                rate = (
                    metric.item_count / metric.elapsed_seconds
                    if metric.elapsed_seconds > 0
                    else 0.0
                )
                
                lines.append(
                    f"{'  Items Processed':<25}"
                    f"{metric.item_count:>10}"
                )
                
                lines.append(
                    f"{'    Processing Rate':<25}"
                    f"{rate:>10.2f} items/sec"
                )
            lines.append("")    
        lines.append("-" * 60)
        lines.append(
            f"{'Total':<25}"
            f"{total:>10.2f} sec"
        )
        lines.append("=" * 60)
        return "\n".join(lines)

    def reset(self) -> None:
        """
        Clear all collected performance metrics.
        """
        self._start_times.clear()
        self._metrics.clear()

    def total_time(self) -> float:
        """
        Return the total elapsed time across all completed stages.
        """
        return sum(
            metric.elapsed_seconds
            for metric in self._metrics.values()
        )

    def print_summary(self) -> None:
        """
        Print the formatted performance summary.
        """
        print(self.summary())