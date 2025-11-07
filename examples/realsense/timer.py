# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from __future__ import annotations

import time
from typing import Literal, Any
import hashlib

import nvtx

# Global object for storing name - > accumulator
_timers = {}


class Accumulator:
    """Holds an an accumulated value + num added elements."""

    def __init__(self) -> None:
        """Constructor."""
        self._num_samples = 0
        self._sum = 0.0
        self._last_value = 0.0

    def mean(self) -> float:
        """Get the mean value."""
        if self._num_samples > 0:
            return self._sum / self._num_samples
        else:
            return -1

    def last(self) -> float | None:
        """Get the most recently added value."""
        return self._last_value

    def num_samples(self) -> int:
        """Get number of samples."""
        return self._num_samples

    def sum(self) -> float:
        """Get the current sum."""
        return self._sum

    def accumulate(self, value: float) -> None:
        """Add a new value to the accumulator."""
        # Note(dtingdahl) This might be inaccurate if _sum is very large.
        self._sum += value
        self._num_samples += 1
        self._last_value = value


def string_to_color(s: str) -> int:
    """Hash the string using SHA-256 and take the first 3 bytes."""
    h = hashlib.sha256(s.encode('utf-8')).digest()
    r, g, b = h[0], h[1], h[2]
    return (r << 16) + (g << 8) + b    # Equivalent to 0xRRGGBB


class Timer:
    """Named timer using perf counter.

    Prefered usage is as a context manager (RAII):

        with Timer("my_function_timer") as _:
            my_function()

    Can also be explicitly started/stopped:

        timer = Timer("my_function_timer")
        my_function()
        timer.stop()
    """

    def __init__(self, name: str):
        """Construct and start a timer."""
        self._name = name
        self._start_time = time.perf_counter()
        self.nvtx_range = nvtx.start_range(message=name, color=string_to_color(name))

    def __enter__(self) -> Timer:
        """Enable use as a context manager."""
        return self

    def stop(self) -> float:
        """Stop the timer and store the result in the global timer object. Returns elapsed time."""
        elapsed = time.perf_counter() - self._start_time
        nvtx.end_range(self.nvtx_range)
        if not self._name in _timers:
            _timers[self._name] = Accumulator()
        _timers[self._name].accumulate(elapsed)
        return elapsed

    def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> Literal[False]:
        """Stop the timer when context manager is released."""
        self.stop()
        return False    # Returning False will cause any exceptions to be propagated


def get_last_time(timer_name: str) -> float | None:
    """Return the last measurement added to timer_name."""
    if timer_name in _timers:
        return _timers[timer_name].last()
    else:
        return 0.0


def get_mean_time(timer_name: str) -> float:
    """Return the mean measurement added to timer_name."""
    if timer_name in _timers:
        return _timers[timer_name].mean()
    else:
        return 0


def timer_status_string() -> str:
    """Return a string containing tabulated status of all timers."""
    if len(_timers.keys()) == 0:
        return ''

    # How much space do we need for the longest timer name?
    name_field_length = max(len(name) for name in _timers) + 2

    string = '\n'
    fmt_name = '{: <' + str(name_field_length) + '}'
    fmt_int = '{: <20}'
    fmt_flt = '{: <20.3}'
    string += fmt_name.format('Timer name') + fmt_int.format('Mean[ms]') + fmt_int.format(
        'Total[s]') + fmt_int.format('Num') + '\n'
    string += '--------------------------------------------------------------------------------\n'
    for (name, accumulator) in sorted(_timers.items()):
        mean_ms = 1000 * accumulator.mean()
        num_samples = accumulator.num_samples()
        string += fmt_name.format(name) + fmt_flt.format(mean_ms) + fmt_flt.format(
            accumulator.sum()) + fmt_int.format(num_samples) + '\n'
    return string