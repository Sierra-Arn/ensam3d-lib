# Copyright (c) 2026 Ilya Snegov (aka Sierra Arn)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/ensam3d_inference/examples/keypoints.py
from typing import NamedTuple


class KeypointInfo(NamedTuple):
    """
    Descriptor for a single semantic keypoint definition.

    Attributes
    ----------
    name : str
        Human-readable keypoint identifier corresponding to the semantic
        joint represented by the model output (e.g. "left_knee").
    color : tuple of int
        RGB visualization color associated with the keypoint, represented
        as (R, G, B) with each channel in the range [0, 255].
    """

    name: str
    color: tuple[int, int, int]


class SkeletonLink(NamedTuple):
    """
    Descriptor for a single skeleton connection between two keypoints.

    Attributes
    ----------
    start : int
        Index of the starting keypoint in KEYPOINTS; corresponds to the
        keypoint index in the model output pred_keypoints_2d.
    end : int
        Index of the ending keypoint in KEYPOINTS; corresponds to the
        keypoint index in the model output pred_keypoints_2d.
    color : tuple of int
        RGB visualization color associated with the link, represented as
        (R, G, B) with each channel in the range [0, 255].
    """

    start: int
    end: int
    color: tuple[int, int, int]


KEYPOINTS: list[KeypointInfo] = [
    KeypointInfo(name="nose",                           color=(51, 153, 255)),
    KeypointInfo(name="left_eye",                       color=(51, 153, 255)),
    KeypointInfo(name="right_eye",                      color=(51, 153, 255)),
    KeypointInfo(name="left_ear",                       color=(51, 153, 255)),
    KeypointInfo(name="right_ear",                      color=(51, 153, 255)),
    KeypointInfo(name="left_shoulder",                  color=(51, 153, 255)),
    KeypointInfo(name="right_shoulder",                 color=(51, 153, 255)),
    KeypointInfo(name="left_elbow",                     color=(51, 153, 255)),
    KeypointInfo(name="right_elbow",                    color=(51, 153, 255)),
    KeypointInfo(name="left_hip",                       color=(51, 153, 255)),
    KeypointInfo(name="right_hip",                      color=(51, 153, 255)),
    KeypointInfo(name="left_knee",                      color=(51, 153, 255)),
    KeypointInfo(name="right_knee",                     color=(51, 153, 255)),
    KeypointInfo(name="left_ankle",                     color=(51, 153, 255)),
    KeypointInfo(name="right_ankle",                    color=(51, 153, 255)),
    KeypointInfo(name="left_big_toe",                   color=(51, 153, 255)),
    KeypointInfo(name="left_small_toe",                 color=(51, 153, 255)),
    KeypointInfo(name="left_heel",                      color=(51, 153, 255)),
    KeypointInfo(name="right_big_toe",                  color=(51, 153, 255)),
    KeypointInfo(name="right_small_toe",                color=(51, 153, 255)),
    KeypointInfo(name="right_heel",                     color=(51, 153, 255)),
    KeypointInfo(name="right_thumb_tip",                color=(51, 153, 255)),
    KeypointInfo(name="right_thumb_first_joint",        color=(51, 153, 255)),
    KeypointInfo(name="right_thumb_second_joint",       color=(51, 153, 255)),
    KeypointInfo(name="right_thumb_third_joint",        color=(51, 153, 255)),
    KeypointInfo(name="right_index_tip",                color=(51, 153, 255)),
    KeypointInfo(name="right_index_first_joint",        color=(51, 153, 255)),
    KeypointInfo(name="right_index_second_joint",       color=(51, 153, 255)),
    KeypointInfo(name="right_index_third_joint",        color=(51, 153, 255)),
    KeypointInfo(name="right_middle_tip",               color=(51, 153, 255)),
    KeypointInfo(name="right_middle_first_joint",       color=(51, 153, 255)),
    KeypointInfo(name="right_middle_second_joint",      color=(51, 153, 255)),
    KeypointInfo(name="right_middle_third_joint",       color=(51, 153, 255)),
    KeypointInfo(name="right_ring_tip",                 color=(51, 153, 255)),
    KeypointInfo(name="right_ring_first_joint",         color=(51, 153, 255)),
    KeypointInfo(name="right_ring_second_joint",        color=(51, 153, 255)),
    KeypointInfo(name="right_ring_third_joint",         color=(51, 153, 255)),
    KeypointInfo(name="right_pinky_tip",                color=(51, 153, 255)),
    KeypointInfo(name="right_pinky_first_joint",        color=(51, 153, 255)),
    KeypointInfo(name="right_pinky_second_joint",       color=(51, 153, 255)),
    KeypointInfo(name="right_pinky_third_joint",        color=(51, 153, 255)),
    KeypointInfo(name="right_wrist",                    color=(51, 153, 255)),
    KeypointInfo(name="left_thumb_tip",                 color=(51, 153, 255)),
    KeypointInfo(name="left_thumb_first_joint",         color=(51, 153, 255)),
    KeypointInfo(name="left_thumb_second_joint",        color=(51, 153, 255)),
    KeypointInfo(name="left_thumb_third_joint",         color=(51, 153, 255)),
    KeypointInfo(name="left_index_tip",                 color=(51, 153, 255)),
    KeypointInfo(name="left_index_first_joint",         color=(51, 153, 255)),
    KeypointInfo(name="left_index_second_joint",        color=(51, 153, 255)),
    KeypointInfo(name="left_index_third_joint",         color=(51, 153, 255)),
    KeypointInfo(name="left_middle_tip",                color=(51, 153, 255)),
    KeypointInfo(name="left_middle_first_joint",        color=(51, 153, 255)),
    KeypointInfo(name="left_middle_second_joint",       color=(51, 153, 255)),
    KeypointInfo(name="left_middle_third_joint",        color=(51, 153, 255)),
    KeypointInfo(name="left_ring_tip",                  color=(51, 153, 255)),
    KeypointInfo(name="left_ring_first_joint",          color=(51, 153, 255)),
    KeypointInfo(name="left_ring_second_joint",         color=(51, 153, 255)),
    KeypointInfo(name="left_ring_third_joint",          color=(51, 153, 255)),
    KeypointInfo(name="left_pinky_tip",                 color=(51, 153, 255)),
    KeypointInfo(name="left_pinky_first_joint",         color=(51, 153, 255)),
    KeypointInfo(name="left_pinky_second_joint",        color=(51, 153, 255)),
    KeypointInfo(name="left_pinky_third_joint",         color=(51, 153, 255)),
    KeypointInfo(name="left_wrist",                     color=(51, 153, 255)),
    KeypointInfo(name="left_olecranon",                 color=(51, 153, 255)),
    KeypointInfo(name="right_olecranon",                color=(51, 153, 255)),
    KeypointInfo(name="left_cubital_fossa",             color=(51, 153, 255)),
    KeypointInfo(name="right_cubital_fossa",            color=(51, 153, 255)),
    KeypointInfo(name="left_acromion",                  color=(51, 153, 255)),
    KeypointInfo(name="right_acromion",                 color=(51, 153, 255)),
    KeypointInfo(name="neck",                           color=(51, 153, 255)),
]
"""
Keypoint descriptors for the 70 MHR body keypoints.

The list index matches the keypoint index in the model output
pred_keypoints_2d — KEYPOINTS[i] describes the keypoint at
pred_keypoints_2d[:, i, :].
"""


SKELETON: list[SkeletonLink] = [
    SkeletonLink(start=13, end=11,  color=(0, 255, 0)),      # left_ankle -> left_knee
    SkeletonLink(start=11, end=9,   color=(0, 255, 0)),      # left_knee -> left_hip
    SkeletonLink(start=14, end=12,  color=(255, 128, 0)),    # right_ankle -> right_knee
    SkeletonLink(start=12, end=10,  color=(255, 128, 0)),    # right_knee -> right_hip
    SkeletonLink(start=9,  end=10,  color=(51, 153, 255)),   # left_hip -> right_hip
    SkeletonLink(start=5,  end=9,   color=(51, 153, 255)),   # left_shoulder -> left_hip
    SkeletonLink(start=6,  end=10,  color=(51, 153, 255)),   # right_shoulder -> right_hip
    SkeletonLink(start=5,  end=6,   color=(51, 153, 255)),   # left_shoulder -> right_shoulder
    SkeletonLink(start=5,  end=7,   color=(0, 255, 0)),      # left_shoulder -> left_elbow
    SkeletonLink(start=6,  end=8,   color=(255, 128, 0)),    # right_shoulder -> right_elbow
    SkeletonLink(start=7,  end=62,  color=(0, 255, 0)),      # left_elbow -> left_wrist
    SkeletonLink(start=8,  end=41,  color=(255, 128, 0)),    # right_elbow -> right_wrist
    SkeletonLink(start=1,  end=2,   color=(51, 153, 255)),   # left_eye -> right_eye
    SkeletonLink(start=0,  end=1,   color=(51, 153, 255)),   # nose -> left_eye
    SkeletonLink(start=0,  end=2,   color=(51, 153, 255)),   # nose -> right_eye
    SkeletonLink(start=1,  end=3,   color=(51, 153, 255)),   # left_eye -> left_ear
    SkeletonLink(start=2,  end=4,   color=(51, 153, 255)),   # right_eye -> right_ear
    SkeletonLink(start=3,  end=5,   color=(51, 153, 255)),   # left_ear -> left_shoulder
    SkeletonLink(start=4,  end=6,   color=(51, 153, 255)),   # right_ear -> right_shoulder
    SkeletonLink(start=13, end=15,  color=(0, 255, 0)),      # left_ankle -> left_big_toe
    SkeletonLink(start=13, end=16,  color=(0, 255, 0)),      # left_ankle -> left_small_toe
    SkeletonLink(start=13, end=17,  color=(0, 255, 0)),      # left_ankle -> left_heel
    SkeletonLink(start=14, end=18,  color=(255, 128, 0)),    # right_ankle -> right_big_toe
    SkeletonLink(start=14, end=19,  color=(255, 128, 0)),    # right_ankle -> right_small_toe
    SkeletonLink(start=14, end=20,  color=(255, 128, 0)),    # right_ankle -> right_heel
    SkeletonLink(start=62, end=45,  color=(255, 128, 0)),    # left_wrist -> left_thumb_third_joint
    SkeletonLink(start=45, end=44,  color=(255, 128, 0)),    # left_thumb_third_joint -> left_thumb_second_joint
    SkeletonLink(start=44, end=43,  color=(255, 128, 0)),    # left_thumb_second_joint -> left_thumb_first_joint
    SkeletonLink(start=43, end=42,  color=(255, 128, 0)),    # left_thumb_first_joint -> left_thumb_tip
    SkeletonLink(start=62, end=49,  color=(255, 153, 255)),  # left_wrist -> left_index_third_joint
    SkeletonLink(start=49, end=48,  color=(255, 153, 255)),  # left_index_third_joint -> left_index_second_joint
    SkeletonLink(start=48, end=47,  color=(255, 153, 255)),  # left_index_second_joint -> left_index_first_joint
    SkeletonLink(start=47, end=46,  color=(255, 153, 255)),  # left_index_first_joint -> left_index_tip
    SkeletonLink(start=62, end=53,  color=(102, 178, 255)),  # left_wrist -> left_middle_third_joint
    SkeletonLink(start=53, end=52,  color=(102, 178, 255)),  # left_middle_third_joint -> left_middle_second_joint
    SkeletonLink(start=52, end=51,  color=(102, 178, 255)),  # left_middle_second_joint -> left_middle_first_joint
    SkeletonLink(start=51, end=50,  color=(102, 178, 255)),  # left_middle_first_joint -> left_middle_tip
    SkeletonLink(start=62, end=57,  color=(255, 51, 51)),    # left_wrist -> left_ring_third_joint
    SkeletonLink(start=57, end=56,  color=(255, 51, 51)),    # left_ring_third_joint -> left_ring_second_joint
    SkeletonLink(start=56, end=55,  color=(255, 51, 51)),    # left_ring_second_joint -> left_ring_first_joint
    SkeletonLink(start=55, end=54,  color=(255, 51, 51)),    # left_ring_first_joint -> left_ring_tip
    SkeletonLink(start=62, end=61,  color=(0, 255, 0)),      # left_wrist -> left_pinky_third_joint
    SkeletonLink(start=61, end=60,  color=(0, 255, 0)),      # left_pinky_third_joint -> left_pinky_second_joint
    SkeletonLink(start=60, end=59,  color=(0, 255, 0)),      # left_pinky_second_joint -> left_pinky_first_joint
    SkeletonLink(start=59, end=58,  color=(0, 255, 0)),      # left_pinky_first_joint -> left_pinky_tip
    SkeletonLink(start=41, end=24,  color=(255, 128, 0)),    # right_wrist -> right_thumb_third_joint
    SkeletonLink(start=24, end=23,  color=(255, 128, 0)),    # right_thumb_third_joint -> right_thumb_second_joint
    SkeletonLink(start=23, end=22,  color=(255, 128, 0)),    # right_thumb_second_joint -> right_thumb_first_joint
    SkeletonLink(start=22, end=21,  color=(255, 128, 0)),    # right_thumb_first_joint -> right_thumb_tip
    SkeletonLink(start=41, end=28,  color=(255, 153, 255)),  # right_wrist -> right_index_third_joint
    SkeletonLink(start=28, end=27,  color=(255, 153, 255)),  # right_index_third_joint -> right_index_second_joint
    SkeletonLink(start=27, end=26,  color=(255, 153, 255)),  # right_index_second_joint -> right_index_first_joint
    SkeletonLink(start=26, end=25,  color=(255, 153, 255)),  # right_index_first_joint -> right_index_tip
    SkeletonLink(start=41, end=32,  color=(102, 178, 255)),  # right_wrist -> right_middle_third_joint
    SkeletonLink(start=32, end=31,  color=(102, 178, 255)),  # right_middle_third_joint -> right_middle_second_joint
    SkeletonLink(start=31, end=30,  color=(102, 178, 255)),  # right_middle_second_joint -> right_middle_first_joint
    SkeletonLink(start=30, end=29,  color=(102, 178, 255)),  # right_middle_first_joint -> right_middle_tip
    SkeletonLink(start=41, end=36,  color=(255, 51, 51)),    # right_wrist -> right_ring_third_joint
    SkeletonLink(start=36, end=35,  color=(255, 51, 51)),    # right_ring_third_joint -> right_ring_second_joint
    SkeletonLink(start=35, end=34,  color=(255, 51, 51)),    # right_ring_second_joint -> right_ring_first_joint
    SkeletonLink(start=34, end=33,  color=(255, 51, 51)),    # right_ring_first_joint -> right_ring_tip
    SkeletonLink(start=41, end=40,  color=(0, 255, 0)),      # right_wrist -> right_pinky_third_joint
    SkeletonLink(start=40, end=39,  color=(0, 255, 0)),      # right_pinky_third_joint -> right_pinky_second_joint
    SkeletonLink(start=39, end=38,  color=(0, 255, 0)),      # right_pinky_second_joint -> right_pinky_first_joint
    SkeletonLink(start=38, end=37,  color=(0, 255, 0)),      # right_pinky_first_joint -> right_pinky_tip
]
"""
Skeleton link descriptors for the 70 MHR body keypoints.

Each link connects two keypoints by their indices in KEYPOINTS and
pred_keypoints_2d. Colors follow the convention from the original
SAM 3D Body pose_info: green for left body, orange for right body,
blue for central connections, and per-finger colors for hands.
"""