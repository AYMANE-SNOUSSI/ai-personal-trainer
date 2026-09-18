"""Turns frame-by-frame results into messages a person can actually read or hear.

Raw form checks flicker: a fault seen on 5 frames (0.2 s) is noise, and a message shown for
0.2 s cannot be read. A fault must persist before it is announced, an announcement stays up long
enough to read, and the same advice is not repeated every second.
"""
from __future__ import annotations

from dataclasses import dataclass

from .engine import FrameResult


@dataclass
class Message:
    text: str
    kind: str          # "good", "fault", "info"
    since: float
    key: str


class LiveCoach:
    def __init__(self, messages: dict[str, str], persist_s: float = 0.5, hold_s: float = 2.5,
                 repeat_after_s: float = 6.0):
        self.messages = messages
        self.persist_s, self.hold_s, self.repeat_after_s = persist_s, hold_s, repeat_after_s
        self._fault_since: dict[str, float] = {}
        self._last_said: dict[str, float] = {}
        self.current: Message | None = None
        self.log: list[Message] = []          # everything announced, e.g. for voice output

    def _announce(self, key: str, text: str, kind: str, t: float) -> Message:
        msg = Message(text, kind, t, key)
        self.current, self._last_said[key] = msg, t
        self.log.append(msg)
        return msg

    def update(self, res: FrameResult) -> Message | None:
        """Returns a message only at the moment it is newly announced; `current` is what to display."""
        t, new = res.t, None
        # a finished rep: short confirmation, or the first problem found in it
        if res.completed is not None:
            rep = res.completed
            key = rep.faults[0] if rep.faults else None
            if key is None:
                new = self._announce(f"rep{rep.index}", f"Good rep. That's {rep.index}.", "good", t)
            elif t - self._last_said.get(key, -1e9) < self.repeat_after_s:
                # advice just given: count the rep without repeating the same sentence
                new = self._announce(f"rep{rep.index}", f"That's {rep.index}.", "info", t)
            else:
                new = self._announce(key, self.messages[key], "fault", t)
        elif res.partial:
            new = self._announce("partial", "That one didn't count. Go all the way down and back up.", "info", t)

        # live faults, only once they persist
        for key in list(self._fault_since):
            if key not in res.live_faults:
                del self._fault_since[key]
        for key in res.live_faults:
            start = self._fault_since.setdefault(key, t)
            recent = t - self._last_said.get(key, -1e9) < self.repeat_after_s
            if new is None and t - start >= self.persist_s and not recent:
                new = self._announce(key, self.messages[key], "fault", t)

        if self.current is not None and t - self.current.since > self.hold_s:
            self.current = None
        return new
