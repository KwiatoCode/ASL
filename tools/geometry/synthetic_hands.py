"""
Syntetyczne szkielety dłoni ze ZNANYMI współrzędnymi – wyłącznie do testowania rekonstrukcji punktów.

Model kinematyczny 3D (palce zginane w stawach MCP/PIP/DIP, kciuk z odwiedzeniem i opozycją, obrót całej dłoni)
→ rzut prostokątny na kadr kamery 640×480 → rysunek identyczny z `mediapipe.solutions.drawing_utils.draw_landmarks`
(czerwone kości grubości 2, zielone punkty r=2 z białą obwódką r=3) na płótnie 400×400 → JPEG.

Tych danych NIE używa się do treningu klasyfikatora: nie odpowiadają literom ASL, tylko sprawdzają, czy rekonstrukcja
poprawnie odzyskuje położenie i numerację stawów, także przy nakładających się punktach.
"""

import io
import math

import cv2
import numpy as np
from PIL import Image

from hand_topology import CONNECTIONS

CAMERA_W, CAMERA_H = 640, 480
CANVAS = 400

_MCP = {'index': (0.30, 0.92), 'middle': (0.08, 0.98), 'ring': (-0.13, 0.93), 'little': (-0.31, 0.82)}
_BASE_ANGLE = {'index': 0.12, 'middle': 0.02, 'ring': -0.08, 'little': -0.20}
_LENGTHS = {'index': (0.42, 0.25, 0.20), 'middle': (0.47, 0.29, 0.22),
            'ring': (0.44, 0.27, 0.21), 'little': (0.34, 0.20, 0.19)}
_ORDER = ['index', 'middle', 'ring', 'little']


def _rotation(roll, pitch, yaw):
    cz, sz = math.cos(roll), math.sin(roll)
    cx, sx = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rz @ rx @ ry


def sample_params(rng):
    """Losowe parametry pozy dłoni (zgięcia palców, rozstaw, kciuk)."""
    style = rng.choice(['open', 'fist', 'mixed', 'mixed', 'mixed'])
    fingers = {}
    for name in _ORDER:
        state = {'open': 'straight', 'fist': 'curled'}.get(style) or rng.choice(['straight', 'half', 'curled'])
        mean = {'straight': [0.1, 0.1, 0.05], 'half': [0.7, 0.9, 0.5], 'curled': [1.4, 1.7, 1.0]}[state]
        fingers[name] = {'flex': rng.normal(mean, 0.15), 'angle_noise': rng.normal(0, 0.04)}
    return {'scale_len': rng.uniform(0.9, 1.1), 'spread': rng.uniform(0.3, 1.6), 'fingers': fingers,
            'abduction': rng.uniform(0.3, 1.4), 'opposition': rng.uniform(0.0, 1.2),
            'thumb_flex': rng.uniform([0.0, 0.0, 0.0], [0.6, 1.2, 1.2])}


def perturb_params(params, rng, amount=0.12):
    """Ta sama „litera” wykonana jeszcze raz: małe zmiany zgięć i ustawienia kciuka."""
    p = {'scale_len': params['scale_len'] * rng.uniform(0.97, 1.03),
         'spread': params['spread'] * rng.uniform(0.9, 1.1),
         'fingers': {k: {'flex': v['flex'] + rng.normal(0, amount, 3), 'angle_noise': v['angle_noise']}
                     for k, v in params['fingers'].items()},
         'abduction': params['abduction'] + rng.normal(0, amount),
         'opposition': params['opposition'] + rng.normal(0, amount),
         'thumb_flex': params['thumb_flex'] + rng.normal(0, amount, 3)}
    return p


def build_hand(params):
    """Punkty 3D dłoni w układzie dłoni: y – wzdłuż palców, x – w poprzek, z – w stronę kamery."""
    points = np.zeros((21, 3))
    scale_len, spread = params['scale_len'], params['spread']
    n = np.array([0.0, 0.0, 1.0])
    for k, name in enumerate(_ORDER):
        mx, my = _MCP[name]
        mcp = np.array([mx, my, 0.0]) * scale_len
        idx = 5 + 4 * k
        points[idx] = mcp
        angle = _BASE_ANGLE[name] * spread + params['fingers'][name]['angle_noise']
        d = np.array([-math.sin(angle), math.cos(angle), 0.0])
        theta, p = 0.0, mcp
        for s, length in enumerate(_LENGTHS[name]):
            theta += params['fingers'][name]['flex'][s]
            p = p + length * scale_len * (math.cos(theta) * d + math.sin(theta) * n)
            points[idx + s + 1] = p
    cmc = np.array([0.22, 0.18, 0.0]) * scale_len
    points[1] = cmc
    d = np.array([math.sin(params['abduction']), math.cos(params['abduction']), 0.0])
    d = math.cos(params['opposition']) * d + math.sin(params['opposition']) * n
    bend_to = np.array([-1.0, 0.2, 0.6])
    bend_to /= np.linalg.norm(bend_to)
    theta, p = 0.0, cmc
    for s, length in enumerate((0.33, 0.28, 0.24)):
        theta += params['thumb_flex'][s]
        direction = math.cos(theta) * d + math.sin(theta) * bend_to
        p = p + length * scale_len * direction / np.linalg.norm(direction)
        points[2 + s] = p
    return points


def hand_3d(rng):
    return build_hand(sample_params(rng))


def sample_view(rng):
    roll = rng.normal(0, 0.45)
    if rng.random() < 0.1:
        roll += rng.choice([-1, 1]) * math.pi / 2
    return {'roll': roll, 'pitch': rng.uniform(-0.6, 0.6), 'yaw': rng.uniform(-0.7, 0.7), 'mirror': rng.random() < 0.5}


def project(points3d, rng, view=None, jitter=0.0):
    """Obrót + rzut na kadr kamery 640×480 (piksele, y w dół). Zwraca znormalizowane współrzędne MediaPipe."""
    view = view or sample_view(rng)
    r = _rotation(view['roll'] + rng.normal(0, jitter), view['pitch'] + rng.normal(0, jitter),
                  view['yaw'] + rng.normal(0, jitter))
    p = points3d @ r.T
    if view['mirror']:
        p[:, 0] = -p[:, 0]                      # druga dłoń / odbicie
    palm_px = rng.uniform(110, 260)
    xy = np.stack([p[:, 0], -p[:, 1]], axis=1) * palm_px
    lo, hi = xy.min(0), xy.max(0)
    span = hi - lo
    ox = rng.uniform(5 - lo[0], max(6 - lo[0], CAMERA_W - 5 - hi[0]))
    oy = rng.uniform(5 - lo[1], max(6 - lo[1], CAMERA_H - 5 - hi[1]))
    xy = xy + [ox, oy]
    if (span > [CAMERA_W - 10, CAMERA_H - 10]).any():
        return None
    return xy / [CAMERA_W, CAMERA_H]


def to_pixel(normalized):
    """`mediapipe.solutions.drawing_utils._normalized_to_pixel_coordinates` dla płótna 400×400."""
    return np.minimum(np.floor(normalized * CANVAS), CANVAS - 1).astype(int)


def draw_like_mediapipe(normalized):
    """Odtworzenie `draw_landmarks`: najpierw wszystkie kości, potem punkty (biała obwódka, zielone koło)."""
    image = np.full((CANVAS, CANVAS, 3), 255, np.uint8)       # RGB
    px = to_pixel(normalized)
    for a, b in CONNECTIONS:
        cv2.line(image, tuple(map(int, px[a])), tuple(map(int, px[b])), (255, 0, 0), 2)
    radius, thickness = 2, 2
    border = max(radius + 1, int(radius * 1.2))
    for p in px:
        cv2.circle(image, tuple(map(int, p)), border, (255, 255, 255), thickness)
        cv2.circle(image, tuple(map(int, p)), radius, (0, 255, 0), thickness)
    return image


def jpeg(image, quality):
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format='JPEG', quality=int(quality))
    return np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB'))


def sample(rng):
    """Jedna próbka: (obraz RGB 400×400, prawdziwe piksele stawów na płótnie 21×2 jako float)."""
    while True:
        normalized = project(hand_3d(rng), rng)
        if normalized is not None and (normalized >= 0).all() and (normalized < 1).all():
            break
    image = jpeg(draw_like_mediapipe(normalized), rng.uniform(75, 95))
    return image, to_pixel(normalized).astype(float)


def letter_dataset(n_letters, per_letter, seed=0):
    """Syntetyczne „litery”: każda ma stałą pozę i ujęcie bazowe, próbki różnią się nieznacznie (jak nagrania
    jednego znaku). Do walidacji etapu 2 (rozstrzyganie na poziomie litery). Zwraca listę (litera, obraz, prawda)."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_letters):
        base, view = sample_params(rng), sample_view(rng)
        view['mirror'] = False
        made = 0
        while made < per_letter:
            normalized = project(build_hand(perturb_params(base, rng)), rng, view, jitter=0.08)
            if normalized is None or not ((normalized >= 0).all() and (normalized < 1).all()):
                continue
            image = jpeg(draw_like_mediapipe(normalized), rng.uniform(75, 95))
            out.append((f'L{k:02d}', image, to_pixel(normalized).astype(float)))
            made += 1
    return out
