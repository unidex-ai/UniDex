import numpy as np

MAPPED_JOINT_DIM = 21

JOINT_MAP = {
    "Inspire": {
        "thumb_proximal_yaw_joint": 0, "thumb_proximal_pitch_joint": 1, "thumb_intermediate_joint": 2, "thumb_distal_joint": 3,
        "index_proximal_joint": 5, "index_intermediate_joint": 7,
        "middle_proximal_joint": 9, "middle_intermediate_joint": 11,
        "ring_proximal_joint": 13, "ring_intermediate_joint": 15,
        "pinky_proximal_joint": 17, "pinky_intermediate_joint": 19
    },
    "Leap": {
        "0": 6, "1": 5, "2": 7, "3": 8,
        "4": 10, "5": 9, "6": 11, "7": 12,
        "8": 14, "9": 13, "10": 15, "11": 16,
        "12": 0, "13": 4, "14": 2, "15": 3, 
    },
    "Wuji": {
        "F1J1": 0, "F1J2": 1, "F1J3": 2, "F1J4": 3,
        "F2J1": 5, "F2J2": 6, "F2J3": 7, "F2J4": 8,
        "F3J1": 9, "F3J2": 10, "F3J3": 11, "F3J4": 12,
        "F4J1": 13, "F4J2": 14, "F4J3": 15, "F4J4": 16,
        "F5J1": 17, "F5J2": 18, "F5J3": 19, "F5J4": 20,
    }
}

RETARGET_JOINT_MAP = {
    "Inspire": {
        "left": {
            "thumb_proximal_yaw_joint": 1, "thumb_proximal_pitch_joint": 1, "thumb_intermediate_joint": 1, "thumb_distal_joint": 1,
            "index_proximal_joint": 1, "index_intermediate_joint": 1,
            "middle_proximal_joint": 1, "middle_intermediate_joint": 1,
            "ring_proximal_joint": 1, "ring_intermediate_joint": 1,
            "pinky_proximal_joint": 1, "pinky_intermediate_joint": 1
        },
        "right": {
            "thumb_proximal_yaw_joint": 1, "thumb_proximal_pitch_joint": 1, "thumb_intermediate_joint": 1, "thumb_distal_joint": 1,
            "index_proximal_joint": 1, "index_intermediate_joint": 1,
            "middle_proximal_joint": 1, "middle_intermediate_joint": 1,
            "ring_proximal_joint": 1, "ring_intermediate_joint": 1,
            "pinky_proximal_joint": 1, "pinky_intermediate_joint": 1
        }
    },
    "Leap": {
        "left": {
            "0": 1, "1": 1, "2": 1, "3": 1,
            "4": 1, "5": 1, "6": 1, "7": 1,
            "8": 1, "9": 1, "10": 1, "11": 1,
            "12": 1, "13": 1, "14": 1, "15": 1
        },
        "right": {
            "0": -1, "1": 1, "2": 1, "3": 1,
            "4": -1, "5": 1, "6": 1, "7": 1,
            "8": -1, "9": 1, "10": 1, "11": 1,
            "12": 1, "13": 1, "14": 1, "15": 1
        }
    },
    "Wuji": {
        "left": {
            "F1J1": 1, "F1J2": 1, "F1J3": 1, "F1J4": 1,
            "F2J1": -1, "F2J2": 1, "F2J3": 1, "F2J4": 1,
            "F3J1": -1, "F3J2": 1, "F3J3": 1, "F3J4": 1,
            "F4J1": -1, "F4J2": 1, "F4J3": 1, "F4J4": 1,
            "F5J1": -1, "F5J2": 1, "F5J3": 1, "F5J4": 1
        },
        "right": {
            "F1J1": 1, "F1J2": 1, "F1J3": 1, "F1J4": 1,
            "F2J1": -1, "F2J2": 1, "F2J3": 1, "F2J4": 1,
            "F3J1": -1, "F3J2": 1, "F3J3": 1, "F3J4": 1,
            "F4J1": -1, "F4J2": 1, "F4J3": 1, "F4J4": 1,
            "F5J1": -1, "F5J2": 1, "F5J3": 1, "F5J4": 1
        }
    }
}

JOINT_DIMENSIONS = {
    "Inspire": 12,
    "Leap": 16,
    "Wuji": 20
}

HAND_TIP_TO_UNIVERSAL = {
    "Inspire": {
        'hand_base_link': 'wrist',
        'thumb_tip': 'thumb_tip',
        'index_tip': 'index_tip',
        'middle_tip': 'middle_tip',
        'ring_tip': 'ring_tip',
        'pinky_tip': 'pinky_tip'
    },
    "Leap": {
        'wrist_link': 'wrist',
        'realtip': 'thumb_tip',
        'realtip_2': 'index_tip',
        'realtip_3': 'middle_tip',
        'realtip_4': 'ring_tip'
    },
    "Wuji": {
        'base_link': 'wrist',
        'fingertip_thumb': 'thumb_tip',
        'fingertip_index': 'index_tip',
        'fingertip_middle': 'middle_tip',
        'fingertip_ring': 'ring_tip',
        'fingertip_pinky': 'pinky_tip'
    }
}

"""
Inverse mapping from real base to virtual base e.g. hand_base_link to base
"""
HAND_TRANSFORMS = {
    "Inspire": {
        "left": np.array([
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ]),
        "right": np.array([
            [-1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
    },
    "Leap": {
        "left": np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        "right": np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    },
    "Wuji": {
        "left": np.array([
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        "right": np.array([
            [0.0, 0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
    }
}

UNIVERSAL_TIPS = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
UNIVERSAL_LINKS = ['wrist', 'thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']

TIPS = {
    "Inspire": ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'],
    "Leap": ['realtip', 'realtip_2', 'realtip_3', 'realtip_4'],
    "Wuji": ['fingertip_thumb', 'fingertip_index', 'fingertip_middle', 'fingertip_ring', 'fingertip_pinky']
}