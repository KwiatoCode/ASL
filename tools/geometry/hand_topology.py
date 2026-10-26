"""Topologia dłoni wspólna dla całego projektu (kolejność MediaPipe = kolejność `HandJoint` w Swift)."""

N_JOINTS = 21

JOINT_NAMES = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_mcp', 'index_pip', 'index_dip', 'index_tip',
    'middle_mcp', 'middle_pip', 'middle_dip', 'middle_tip',
    'ring_mcp', 'ring_pip', 'ring_dip', 'ring_tip',
    'little_mcp', 'little_pip', 'little_dip', 'little_tip',
]

# mediapipe.solutions.hands.HAND_CONNECTIONS – kości, którymi rysowano dataset.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

FINGERS = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'little': [17, 18, 19, 20],
}
TIPS = [4, 8, 12, 16, 20]
MCPS = [5, 9, 13, 17]
