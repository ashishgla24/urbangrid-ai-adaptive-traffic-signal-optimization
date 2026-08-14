"""
A real adaptive signal controller: a state machine with phase memory,
minimum/maximum green times, mandatory all-red clearance, and
starvation prevention — instead of picking a "winner lane" fresh
every single video frame (the old behaviour), which:
  1) flickers, since the busiest lane can change frame to frame
  2) has no concept of "time already spent green"
  3) can starve low-traffic lanes indefinitely

This is still a heuristic controller (not RL/optimization-based),
but it is at least a *correct* one you could run on real hardware.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

from config import CONFIG


class Phase(Enum):
    GREEN = auto()
    ALL_RED_CLEARANCE = auto()


@dataclass
class SignalState:
    active_lane: int = 0
    phase: Phase = Phase.GREEN
    phase_started_at: float = field(default_factory=time.time)
    green_duration: float = CONFIG.signal.min_green_sec
    last_green_at: List[float] = field(
        default_factory=lambda: [time.time()] * CONFIG.signal.num_lanes
    )


class AdaptiveSignalController:
    """Call `update(lane_counts)` once per control tick (e.g. every
    1-2 seconds, using *smoothed* counts — not raw per-frame counts,
    which are noisy). Returns the current SignalState."""

    def __init__(self, cfg=None):
        self.cfg = cfg or CONFIG.signal
        self.state = SignalState(green_duration=self.cfg.min_green_sec)

    def _elapsed_in_phase(self) -> float:
        return time.time() - self.state.phase_started_at

    def _pick_next_lane(self, lane_counts: List[int]) -> int:
        """Choose the busiest lane, but force a switch to any lane
        that has been waiting too long (anti-starvation), even if
        it's currently quiet."""
        now = time.time()
        starved = [
            i for i, last in enumerate(self.state.last_green_at)
            if now - last > self.cfg.max_wait_before_forced_switch_sec
        ]
        if starved:
            # serve the most-starved lane first, regardless of current count
            return max(starved, key=lambda i: now - self.state.last_green_at[i])
        return max(range(len(lane_counts)), key=lambda i: lane_counts[i])

    def _green_time_for(self, lane_counts: List[int], lane: int) -> float:
        extra = lane_counts[lane] * self.cfg.vehicles_per_extra_second
        return min(self.cfg.min_green_sec + extra, self.cfg.max_green_sec)

    def update(self, lane_counts: List[int]) -> SignalState:
        elapsed = self._elapsed_in_phase()

        if self.state.phase == Phase.GREEN:
            if elapsed < self.cfg.min_green_sec:
                return self.state  # never cut a phase below the legal minimum

            must_switch = elapsed >= self.state.green_duration
            if must_switch:
                self.state.phase = Phase.ALL_RED_CLEARANCE
                self.state.phase_started_at = time.time()

        elif self.state.phase == Phase.ALL_RED_CLEARANCE:
            if elapsed >= self.cfg.all_red_clearance_sec:
                next_lane = self._pick_next_lane(lane_counts)
                self.state.active_lane = next_lane
                self.state.green_duration = self._green_time_for(lane_counts, next_lane)
                self.state.last_green_at[next_lane] = time.time()
                self.state.phase = Phase.GREEN
                self.state.phase_started_at = time.time()

        return self.state

    def force_emergency_green(self, lane: int, duration: float = None):
        """Preempt the normal cycle for an emergency vehicle. Skips
        clearance-interval logic deliberately (real controllers use a
        short mandatory all-red before the emergency phase too — add
        that here if your hardware requires it)."""
        self.state.active_lane = lane
        self.state.green_duration = duration or self.cfg.max_green_sec
        self.state.last_green_at[lane] = time.time()
        self.state.phase = Phase.GREEN
        self.state.phase_started_at = time.time()