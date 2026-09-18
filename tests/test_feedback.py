from coach import EXERCISES, RepCounter
from coach.engine import FrameResult
from coach.feedback import LiveCoach


def frame(t, faults=()):
    return FrameResult(t=t, visible=True, side="left", angle=150, phase="top", reps=0, live_faults=list(faults))


def coach():
    return LiveCoach(RepCounter(EXERCISES["pushup"]).messages())


def test_short_flicker_is_not_announced():
    c = coach()
    for i in range(5):                                 # 0.2 s of fault at 25 fps
        c.update(frame(i / 25, ["hips_piked"]))
    for i in range(5, 50):
        c.update(frame(i / 25))
    assert c.log == []


def test_persistent_fault_is_announced_once_and_held():
    c = coach()
    for i in range(0, 75):                             # 3 s of the same fault
        c.update(frame(i / 25, ["hips_sagging"]))
    assert [m.key for m in c.log] == ["hips_sagging"]
    assert c.current is None or c.current.key == "hips_sagging"


def test_same_advice_is_repeated_only_after_a_while():
    c = coach()
    for i in range(0, 25 * 8):                         # 8 s continuous fault, repeat_after = 6 s
        c.update(frame(i / 25, ["hips_sagging"]))
    assert len(c.log) == 2


def test_message_disappears_after_hold_time():
    c = coach()
    for i in range(0, 20):
        c.update(frame(i / 25, ["hips_piked"]))
    assert c.current is not None
    for i in range(20, 120):
        c.update(frame(i / 25))
    assert c.current is None


def test_rep_with_a_fault_just_announced_does_not_repeat_the_advice():
    from coach.engine import RepRecord
    c = coach()
    for i in range(0, 25):
        c.update(frame(i / 25, ["hips_sagging"]))
    done = FrameResult(t=1.2, visible=True, side="left", angle=150, phase="top", reps=1,
                       completed=RepRecord(1, 0.0, 1.2, 80.0, ["hips_sagging"]))
    c.update(done)
    assert [m.text for m in c.log][-1] == "That's 1."
