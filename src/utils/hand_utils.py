import numpy as np
import json
from pathlib import Path

_HAND_UTILS_JSON_PATH = Path(__file__).parent.parent / "assets" / "utils" / "hand_utils.json"

def _load_hand_utils():
    try:
        with open(_HAND_UTILS_JSON_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"hand_utils.json not found at {_HAND_UTILS_JSON_PATH}")

_hand_utils_data = _load_hand_utils()

MAPPED_JOINT_DIM = _hand_utils_data["mapped_joint_dim"]
JOINT_DIM_IN_USE = _hand_utils_data["joint_dim_in_use"]
UNIVERSAL_TIPS = _hand_utils_data["universal_tips"]
UNIVERSAL_LINKS = _hand_utils_data["universal_links"]
JOINT_DIMENSIONS = _hand_utils_data["joint_dimensions"]
JOINT_MAP = _hand_utils_data["joint_map"]
RETARGET_JOINT_MAP_SCALE = _hand_utils_data["retarget_joint_map_scale"]
RETARGET_JOINT_MAP_OFFSET = _hand_utils_data["retarget_joint_map_offset"]
HAND_TIP_TO_UNIVERSAL = _hand_utils_data["hand_tip_to_universal"]
TIPS = _hand_utils_data["tips"]
JOINTS = _hand_utils_data["joints"]

HAND_TRANSFORMS = {}
for hand in _hand_utils_data["hand_transforms"]:
    HAND_TRANSFORMS[hand] = {}
    for side in _hand_utils_data["hand_transforms"][hand]:
        HAND_TRANSFORMS[hand][side] = np.array(_hand_utils_data["hand_transforms"][hand][side])
