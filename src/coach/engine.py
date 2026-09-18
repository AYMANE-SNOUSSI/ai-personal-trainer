"""Repetition counting and form evaluation over time."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .exercises import ExerciseSpec
from .geometry import joint_angle
from .pose import SidePose, pick_side


@dataclass
class RepRecord:
    index: int
    start: float
    end: float
    min_angle: float
    faults: list[str]
    observations: dict = field(default_factory=dict)   # per measurement: min, median, max over the rep

    @property
    def clean(self) -> bool:
        return not self.faults


@dataclass
class FrameResult:
    t: float
    visible: bool
    side: str | None
    angle: float | None
    phase: str                     # "waiting", "top", "descending", "bottom"
    reps: int
    live_faults: list[str] = field(default_factory=list)
    observations: dict = field(default_factory=dict)
    completed: RepRecord | None = None
    partial: bool = False          # a movement started and was abandoned before the bottom


class RepCounter:
    """Feed one frame of keypoints at a time; get the state of the exercise back.

    Counting starts once the top position has been seen. A rep is: leave the top (angle < up_above), reach the bottom (angle < down_below), return to
    the top (angle > up_above). Every completed rep is counted; form problems are attached to it
    rather than silently discarding it.
    """

    def __init__(self, spec: ExerciseSpec, min_conf: float = 0.5, smoothing: float = 0.5,
                 min_rep_seconds: float = 0.4, max_gap_seconds: float = 1.0,
                 adaptive: bool = True, bottom_fraction: float = 0.3, top_fraction: float = 0.7):
        self.spec, self.min_conf, self.alpha = spec, min_conf, smoothing
        self.min_rep_s, self.max_gap_s = min_rep_seconds, max_gap_seconds
        # Fixed angles never fit everyone: body proportions, camera angle and the detector itself
        # shift them (a fully extended arm was measured at 141 degrees on one clip and 175 on
        # another). The thresholds are therefore placed inside the range this person actually
        # produces, and the fixed ones in the spec are only used until that range is known.
        self.adaptive, self.bottom_fraction, self.top_fraction = adaptive, bottom_fraction, top_fraction
        self.seen_min, self.seen_max = np.inf, -np.inf
        self.reps: list[RepRecord] = []
        self.partial_reps = 0
        self._side: str | None = None
        self._angle: float | None = None
        self._last_seen: float | None = None
        self._started = False          # no coaching until the first rep begins (setting up is not a fault)
        self._reset_rep(phase="waiting")

    def _reset_rep(self, phase: str = "top") -> None:
        # "waiting": the top position must be seen before any rep can start, so a person who
        # appears mid-movement (start of video, after losing tracking) is not half-counted.
        self.phase = phase
        self._rep_start = 0.0
        self._min_angle = np.inf
        self._frames = 0
        self._fault_frames = {r.name: 0 for r in self.spec.rules}
        self._obs: dict[str, list[float]] = {k: [] for k in self.spec.observations}

    def update(self, keypoints: np.ndarray | None, t: float) -> FrameResult:
        """keypoints: array (17, 3) of x, y, confidence for the tracked person, or None."""
        if keypoints is not None:
            self._side = pick_side(keypoints, self.spec.joints, self._side)
            pose = SidePose(keypoints, self._side, self.min_conf)
            raw = self._primary_angle(pose)
        else:
            pose, raw = None, None

        if raw is None:
            # lost the body: tolerate short gaps, abandon the rep on long ones
            if self._last_seen is not None and t - self._last_seen > self.max_gap_s and self.phase != "waiting":
                self._reset_rep(phase="waiting")
                self._angle = None
            return FrameResult(t, False, self._side, None, self.phase, len(self.reps))

        self._last_seen = t
        self._angle = raw if self._angle is None else self.alpha * raw + (1 - self.alpha) * self._angle
        angle = self._angle
        self.seen_min, self.seen_max = min(self.seen_min, angle), max(self.seen_max, angle)
        down_below, up_above = self.thresholds()
        completed, partial = None, False

        if self.phase == "waiting" and angle >= up_above:
            self.phase = "top"
        if self.phase == "top":
            if angle < up_above:
                self.phase, self._rep_start, self._started = "descending", t, True
        live = self._live_faults(pose) if self._started else []
        obs = {}
        for name, metric in self.spec.observations.items():
            v = metric(pose)
            obs[name] = None if v is None or np.isnan(v) else float(v)
        if self.phase in ("descending", "bottom"):
            self._frames += 1
            self._min_angle = min(self._min_angle, angle)
            for name in live:
                self._fault_frames[name] += 1
            for name, v in obs.items():
                if v is not None:
                    self._obs[name].append(v)
            if angle < down_below:
                self.phase = "bottom"
            elif angle >= up_above:
                if self.phase == "bottom" and t - self._rep_start >= self.min_rep_s:
                    completed = self._close_rep(t)
                elif self.phase == "descending" and self._min_angle < up_above - 15:
                    # a real attempt that stopped short; smaller dips are just wobble at the top
                    self.partial_reps += 1
                    partial = True
                self._reset_rep()

        return FrameResult(t, True, self._side, angle, self.phase, len(self.reps), live, obs, completed, partial)

    def thresholds(self) -> tuple[float, float]:
        """Bottom and top thresholds: inside the observed range once it is wide enough."""
        span = self.seen_max - self.seen_min
        if not self.adaptive or not np.isfinite(span) or span < self.spec.min_range:
            return self.spec.down_below, self.spec.up_above
        return (self.seen_min + self.bottom_fraction * span,
                self.seen_min + self.top_fraction * span)

    def _primary_angle(self, pose: SidePose) -> float | None:
        pts = pose.points(*self.spec.primary)     # 3D when the detector estimates depth
        if pts is None:
            return None
        v = joint_angle(*pts)
        return None if np.isnan(v) else v

    def _live_faults(self, pose: SidePose) -> list[str]:
        out = []
        for rule in self.spec.rules:
            value = rule.metric(pose)
            if value is not None and rule.is_fault(value):
                out.append(rule.name)
        return out

    def _close_rep(self, t: float) -> RepRecord:
        stats = {k: {"min": round(float(np.min(v)), 1), "median": round(float(np.median(v)), 1),
                     "max": round(float(np.max(v)), 1)} for k, v in self._obs.items() if v}
        faults = [r.name for r in self.spec.rules
                  if self._frames and self._fault_frames[r.name] / self._frames >= r.min_fraction]
        faults += [r.name for r in self.spec.rep_rules
                   if r.observation in stats and r.is_fault(stats[r.observation])]
        if self.spec.depth_target is not None and self._min_angle > self.spec.depth_target:
            faults.insert(0, "not_deep_enough")
        rep = RepRecord(len(self.reps) + 1, self._rep_start, t, float(self._min_angle), faults, stats)
        self.reps.append(rep)
        return rep

    def messages(self) -> dict[str, str]:
        m = {r.name: r.message for r in self.spec.rules}
        m |= {r.name: r.message for r in self.spec.rep_rules}
        m["not_deep_enough"] = self.spec.depth_message
        return m

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for rep in self.reps:
            for f in rep.faults:
                counts[f] = counts.get(f, 0) + 1
        return {
            "exercise": self.spec.name,
            "reps": len(self.reps),
            "clean_reps": sum(r.clean for r in self.reps),
            "partial_reps": self.partial_reps,
            "angle_range_seen": None if not np.isfinite(self.seen_max - self.seen_min) else
                                [round(float(self.seen_min), 1), round(float(self.seen_max), 1)],
            "thresholds": [round(v, 1) for v in self.thresholds()],
            "faults": counts,
        }
