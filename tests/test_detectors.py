from types import SimpleNamespace

import numpy as np

from detectors import MEDIAPIPE_TO_COCO, landmarks_to_coco


def test_mediapipe_landmarks_map_to_coco_order_in_pixels():
    lms = [SimpleNamespace(x=i / 100, y=i / 200, visibility=0.5 + i / 100) for i in range(33)]
    kp = landmarks_to_coco(lms, width=1000, height=800)
    assert kp.shape == (17, 3)
    # COCO 5 = left shoulder = MediaPipe 11; COCO 16 = right ankle = MediaPipe 28
    assert np.allclose(kp[5], [110, 44, 0.61])
    assert np.allclose(kp[16], [280, 112, 0.78])
    assert len(set(MEDIAPIPE_TO_COCO)) == 17


def test_keypoints_outside_the_image_are_counted():
    import numpy as np

    from detectors import joints_outside_frame

    kp = np.zeros((17, 3))
    kp[5] = [100, 100, 0.9]        # inside
    kp[9] = [110, -12, 0.9]        # hand above the top edge, still reported as visible
    kp[10] = [700, 50, 0.2]        # outside but not confident: not counted
    assert joints_outside_frame(kp, (480, 640, 3)) == 1
    assert joints_outside_frame(None, (480, 640, 3)) == 0


def test_largest_person_is_the_one_kept():
    import numpy as np

    from detectors import body_size

    near, far = np.zeros((17, 3)), np.zeros((17, 3))
    near[[5, 11]] = [[100, 100, 0.9], [110, 400, 0.9]]
    far[[5, 11]] = [[600, 100, 0.9], [604, 160, 0.9]]
    assert body_size(near) > body_size(far)
    assert max([far, near], key=body_size) is near


def test_front_view_is_told_apart_from_a_side_view():
    import numpy as np

    from detectors import facing_camera

    side, front = np.zeros((17, 3)), np.zeros((17, 3))
    side[[5, 6, 11]] = [[300, 100, 0.9], [310, 102, 0.9], [300, 300, 0.9]]     # shoulders overlap
    front[[5, 6, 11]] = [[250, 100, 0.9], [400, 100, 0.9], [250, 300, 0.9]]    # shoulders apart
    assert facing_camera(side) is False
    assert facing_camera(front) is True
    assert facing_camera(np.zeros((17, 3))) is None
    assert facing_camera(None) is None
