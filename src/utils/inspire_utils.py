MIMIC_RELATION = {
    "thumb_intermediate_joint": ["thumb_proximal_pitch_joint", 1.334, 0],
    "thumb_distal_joint": ["thumb_proximal_pitch_joint", 0.667, 0],
    "index_intermediate_joint": ["index_proximal_joint", 1.06399, -0.04545],
    "middle_intermediate_joint": ["middle_proximal_joint", 1.06399, -0.04545],
    "ring_intermediate_joint": ["ring_proximal_joint", 1.06399, -0.04545],
    "pinky_intermediate_joint": ["pinky_proximal_joint", 1.06399, -0.04545]
}

INSPIRE_JOINTS = [
    "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint", "thumb_intermediate_joint", "thumb_distal_joint",
    "index_proximal_joint", "index_intermediate_joint",
    "middle_proximal_joint", "middle_intermediate_joint",
    "ring_proximal_joint", "ring_intermediate_joint",
    "pinky_proximal_joint", "pinky_intermediate_joint"
]

PROXIMAL_JOINTS = list(reversed([
    "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint"
]))