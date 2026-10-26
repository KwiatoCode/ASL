"""
Rekonstrukcja 21 punktów dłoni z obrazów datasetu „ASL Hand Skeleton” (usifelshafie, Kaggle).

Dataset zawiera WYŁĄCZNIE obrazy JPG (26 000 plików, brak CSV ze współrzędnymi). Obrazy narysowano funkcją
`mediapipe draw_landmarks`: najpierw 21 czerwonych kości, potem 21 punktów (biała obwódka + zielone koło).
Rekonstrukcja odwraca ten rysunek:

  1. maski kolorów: zielone punkty, czerwone kości (progi odporne na JPEG),
  2. zielone składowe spójne → kandydaci na punkty; duże/wydłużone składowe dzielone (punkty częściowo nałożone),
  3. dla każdej pary kandydatów: jaka część odcinka między nimi jest pokryta czerwoną kością
     (z pominięciem okolic punktów, gdzie biała obwódka zamazuje kość),
  4. przeszukiwanie wiązkowe przypisań staw → kandydat zgodne z topologią dłoni (21 kości MediaPipe);
     staw schowany pod innym punktem może współdzielić kandydata (kara + flaga „ukryty”),
  5. ocena globalna: jaka część czerwonych pikseli jest wyjaśniona przez 21 kości rozwiązania,
  6. symetria topologii (wskazujący ↔ mały palec) rozstrzygana geometrią kciuka,
  7. decyzja: przyjęte / odrzucone z powodem. Niepewnych próbek NIE uzupełnia się zgadywaniem.

Użycie (Windows, PowerShell):
  python reconstruct_skeleton.py --dataset ..\\..\\..\\asl_dataset\\dataset --out landmarks.csv --jobs 4
  python reconstruct_skeleton.py --dataset <folder> --out landmarks.csv --qa qa_rekonstrukcji   # + podglądy

Wynik CSV: letter, file, status, reason, metryki jakości, x0,y0…x20,y20 (piksele płótna 400×400),
hidden0…hidden20 (1 = staw ukryty pod innym punktem, położenie przybliżone).
"""

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from os.path import basename, join

import cv2
import numpy as np

from hand_topology import CONNECTIONS, N_JOINTS

RECONSTRUCTION_VERSION = 1

# Kolejność przypisywania stawów i rodzic w drzewie rozpinającym (kość, która musi istnieć).
# Najpierw śródręcze (0, 5, 9, 13, 17) – po nim znana jest skala dłoni i można porównywać z szablonem litery.
ORDER = [0, 5, 9, 13, 17, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
PARENT = {1: 0, 2: 1, 3: 2, 4: 3, 5: 0, 6: 5, 7: 6, 8: 7, 9: 5, 10: 9, 11: 10, 12: 11,
          13: 9, 14: 13, 15: 14, 16: 15, 17: 13, 18: 17, 19: 18, 20: 19}
PALM = [0, 5, 9, 13, 17]
ASPECT_X = 640 / 480   # płótno 400×400 powstało z kadru 4:3 – oś X jest ściśnięta (README, rozdz. 2)
EXTRA_EDGES = {17: [0]}          # kość zamykająca śródręcze 0–17
# Automorfizm topologii: odbicie dłoni zamienia palec wskazujący z małym i środkowy z serdecznym.
TWIN = [0, 1, 2, 3, 4, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8]

BEAM = 1000                     # szerokość wiązki przeszukiwania
MIN_EDGE = 0.55                 # minimalne pokrycie odcinka czerwoną kością, by uznać go za kość
REUSE_PENALTY = math.log(0.02)  # staw schowany pod innym punktem (współdzielony kandydat)
EXCLUDE_RADIUS = 4.5            # biała obwódka punktu (r = 3, grubość 2) zamazuje kość w tym promieniu
EXPLAIN_WEIGHT = 100.0          # waga zgodności narysowanego rozwiązania z czerwonymi pikselami
MIN_MARGIN = 3.0                # przewaga najlepszego rozwiązania nad najlepszym INNYM (inne położenia stawów)
TURN_PENALTY = 0.0             # zawracanie palca w rzucie 2D jest częste przy zgiętych palcach – bez kary (rozstrzyga etap 2)
SIDE_PENALTY = math.log(0.01)   # podstawa kciuka po stronie małego palca – numeracja odbita
THUMB_PENALTY = math.log(0.3)   # numeracja z kciukiem bliżej małego palca – możliwa (pięść), ale mniej prawdopodobna
RESCORE = 150                   # ile najlepszych pełnych rozwiązań ocenić globalnie
TEMPLATE_WEIGHT = 1.0           # waga szablonu kształtu litery (etap 2)
ORDER_PENALTY = math.log(0.02)  # naruszona kolejność rysowania nakładających się punktów
N_ALTERNATIVES = 6              # ile różnych rozwiązań zapisać dla etapu 2


# ---------------------------------------------------------------- detekcja kolorów i kandydatów

def color_masks(rgb):
    img = rgb.astype(np.int16)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    green = (g > 140) & (r < 140) & (b < 140) & (g - r > 60) & (g - b > 60)
    red = (r > 140) & (g < 130) & (b < 130) & (r - g > 60) & (r - b > 60)
    return green, red


def candidates(green):
    """Środki zielonych punktów. Składowe większe niż jeden punkt dzielone k-means na k punktów."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(green.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= 4]
    if not keep:
        return np.zeros((0, 2)), [], 0.0
    single = float(np.median([stats[i, cv2.CC_STAT_AREA] for i in keep]))
    single = min(max(single, 15.0), 40.0)
    points, merged = [], []
    for i in keep:
        area = stats[i, cv2.CC_STAT_AREA]
        ys, xs = np.nonzero(labels == i)
        coords = np.stack([xs, ys], 1).astype(np.float32)
        cov = np.cov(coords.T) if len(coords) > 2 else np.eye(2)
        evals = np.sort(np.linalg.eigvalsh(cov))
        elong = math.sqrt(max(evals[1], 1e-6) / max(evals[0], 1e-6))
        k = 1
        if area > 1.45 * single or elong > 1.8:
            k = max(2, int(round(area / (0.8 * single))))
            k = min(k, 4)
        if k == 1:
            points.append(coords.mean(0))
            merged.append(False)
        else:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            _, _, centers = cv2.kmeans(coords, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            for c in centers:
                points.append(c)
                merged.append(True)
    return np.array(points, dtype=float), merged, single


def drawing_order_pairs(points, green):
    """Pary (wcześniej, później) narysowanych punktów. MediaPipe rysuje punkty po kolei 0…20, a biała obwódka
    późniejszego punktu nadgryza wcześniejszy – przy odległości < 6.5 px widać, który jest „na wierzchu”.
    Na syntetycznych rysunkach: 97–100 % trafności dla par bliższych niż 6 px."""
    yy, xx = np.mgrid[-4:5, -4:5]
    disk = (xx ** 2 + yy ** 2) <= 2.5 ** 2 + 0.1
    h, w = green.shape
    completeness = []
    for p in points:
        x, y = int(round(p[0])), int(round(p[1]))
        if 4 <= x < w - 4 and 4 <= y < h - 4:
            completeness.append(green[y - 4:y + 5, x - 4:x + 5][disk].mean())
        else:
            completeness.append(np.nan)
    pairs = []
    for a in range(len(points)):
        for b in range(a + 1, len(points)):
            if np.hypot(*(points[a] - points[b])) < 6.5 and abs(completeness[a] - completeness[b]) > 0.1:
                pairs.append((a, b) if completeness[a] < completeness[b] else (b, a))
    return pairs


def order_penalty(assign, pairs):
    """Kara za każdą parę, w której punkt „pod spodem” dostał wyższy numer stawu niż punkt „na wierzchu”."""
    if not pairs:
        return 0.0
    top = {}
    for joint, cand in enumerate(assign):
        top[cand] = max(top.get(cand, -1), joint)
    bad = sum(1 for under, over in pairs if under in top and over in top and top[under] > top[over])
    return bad * ORDER_PENALTY


def edge_scores(points, red):
    """S[i, j] = pokrycie odcinka i–j czerwoną kością (0…1); NaN gdy odcinek prawie w całości schowany pod punktami."""
    red_d = cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    h, w = red.shape
    n = len(points)
    s = np.full((n, n), 0.0)
    cov = np.full((n, n), 0.0)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = points[i], points[j]
            length = float(np.hypot(*(b - a)))
            steps = max(int(length), 1)
            t = np.linspace(0, 1, steps + 1)
            samples = a + (b - a) * t[:, None]
            d = np.min(np.hypot(samples[:, None, 0] - points[None, :, 0],
                                samples[:, None, 1] - points[None, :, 1]), axis=1)
            eligible = d > EXCLUDE_RADIUS
            if eligible.sum() < 2:
                value = np.nan
            else:
                xs = np.clip(np.round(samples[eligible, 0]).astype(int), 0, w - 1)
                ys = np.clip(np.round(samples[eligible, 1]).astype(int), 0, h - 1)
                value = float(red_d[ys, xs].mean())
            cov[i, j] = cov[j, i] = value
            if not np.isnan(value):
                # Kość MediaPipe łączy SĄSIEDNIE stawy – odcinek przechodzący przez inny punkt to zwykle
                # „przeskok” po łańcuchu palca (A–C przez B), więc dostaje karę.
                ab = b - a
                t_proj = ((points - a) @ ab) / max(length ** 2, 1e-6)
                inside = (t_proj > 0.08) & (t_proj < 0.92)
                inside[[i, j]] = False
                if inside.any():
                    dist = np.hypot(*(points[inside] - (a + np.outer(t_proj[inside], ab))).T).min()
                    if dist < 3:
                        value *= 0.6
                    elif dist < 5:
                        value *= 0.85
            s[i, j] = s[j, i] = value
    return s, cov


# ---------------------------------------------------------------- dopasowanie topologii

def _edge_log(s, a, b):
    if a == b:
        return REUSE_PENALTY
    v = s[a, b]
    if np.isnan(v):
        return math.log(0.6)           # kość krótsza niż obwódki punktów – nie da się jej zmierzyć
    return math.log(max(v, 0.02))


def _turn_penalty(points, assign, joint, b):
    """Kara za zawrócenie łańcucha palca: kąt między kolejnymi kośćmi > ~120° (w tym zawracanie po tej samej linii)."""
    parent = PARENT[joint]
    grand = PARENT.get(parent)
    if grand is None or joint in (5, 9, 13, 17):
        return 0.0
    if parent in (5, 9, 13) and joint in (9, 13, 17):
        return 0.0
    u = points[assign[parent]] - points[assign[grand]]
    v = points[b] - points[assign[parent]]
    nu, nv = np.hypot(*u), np.hypot(*v)
    if nu < 3 or nv < 3:
        return 0.0
    cos = float(u @ v / (nu * nv))
    return TURN_PENALTY if cos < -0.5 else 0.0


def _normalizer(points, assign):
    """Początek = nadgarstek, skala = średnia odległość nadgarstek → MCP (X rozciągnięte do proporcji kadru)."""
    p = points[[assign[j] for j in PALM]].astype(float) * [ASPECT_X, 1.0]
    scale = max(float(np.mean(np.hypot(*(p[1:] - p[0]).T))), 1.0)
    return p[0], scale


def _template_term(points, assign, joint, template, norm):
    if template is None:
        return 0.0
    origin, scale = norm
    q = (points[assign[joint]] * [ASPECT_X, 1.0] - origin) / scale
    d = float(np.hypot(*(q - template['mean'][joint]))) / template['sd'][joint]
    return TEMPLATE_WEIGHT * max(-0.5 * d * d, -8.0)


def beam_search(s, cov, optional, points, template=None):
    """Przypisania staw → kandydat. `optional[k]` = kandydat z podziału zlanej plamy (może zostać nieużyty).
    `template` (opcjonalnie) – szablon kształtu litery z etapu 2, kieruje przeszukiwaniem."""
    n = len(s)
    ok = (cov >= MIN_EDGE) | np.isnan(cov)
    np.fill_diagonal(ok, False)
    neighbours = [np.nonzero(ok[a])[0].tolist() for a in range(n)]
    required = frozenset(k for k in range(n) if not optional[k])
    empty = (-1,) * 21
    beams = [(empty[:0] + (a,) + empty[1:], frozenset([a]), 0.0, None) for a in range(n)]
    for step, joint in enumerate(ORDER[1:], start=1):
        remaining = 21 - step - 1
        parent = PARENT[joint]
        expanded = []
        for assign, used, score, norm in beams:
            a = assign[parent]
            for b in neighbours[a] + [a]:
                new_used = used | {b}
                if len(required - new_used) > remaining:
                    continue
                gain = _edge_log(s, a, b)
                if b in used and b != a:
                    gain += REUSE_PENALTY
                gain += _turn_penalty(points, assign, joint, b)
                new_assign = assign[:joint] + (b,) + assign[joint + 1:]
                for extra in EXTRA_EDGES.get(joint, []):
                    c = assign[extra]
                    if c != b and not ok[b, c]:
                        gain = -math.inf
                        break
                    gain += _edge_log(s, b, c)
                if gain == -math.inf:
                    continue
                new_norm = norm
                if template is not None:
                    if joint == PALM[-1]:
                        new_norm = _normalizer(points, new_assign)
                        gain += sum(_template_term(points, new_assign, j, template, new_norm) for j in PALM)
                    elif new_norm is not None:
                        gain += _template_term(points, new_assign, joint, template, new_norm)
                expanded.append((new_assign, new_used, score + gain, new_norm))
        if not expanded:
            return []
        expanded.sort(key=lambda e: -e[2])
        beams = expanded[:BEAM]
    return [np.array(a) for a, u, sc, nm in beams if required <= u][:RESCORE]


def local_score(s, assign, points, template=None):
    """Wynik lokalny pełnego przypisania: pokrycie kości + kary za współdzielone punkty + szablon litery."""
    total = 0.0
    for joint in ORDER[1:]:
        a, b = assign[PARENT[joint]], assign[joint]
        total += _edge_log(s, a, b)
        total += _turn_penalty(points, assign, joint, b)
    total += _edge_log(s, assign[17], assign[0])
    total += REUSE_PENALTY * (21 - len(set(assign.tolist())))
    if template is not None:
        norm = _normalizer(points, assign)
        total += sum(_template_term(points, assign, j, template, norm) for j in range(21))
    return total


def explained_red(points, assign, red, red_wide=None):
    """Zgodność narysowanego rozwiązania z czerwonymi pikselami obrazu: recall × precision.
    recall – część czerwonych pikseli leżących na kościach rozwiązania,
    precision – część pikseli kości rozwiązania (poza obwódkami punktów), które są czerwone."""
    p = np.round(points[assign]).astype(int)
    wide = np.zeros(red.shape, np.uint8)
    thin = np.zeros(red.shape, np.uint8)
    for a, b in CONNECTIONS:
        cv2.line(wide, tuple(map(int, p[a])), tuple(map(int, p[b])), 1, 5)
        cv2.line(thin, tuple(map(int, p[a])), tuple(map(int, p[b])), 1, 1)
    for q in points:
        cv2.circle(thin, (int(round(q[0])), int(round(q[1]))), 5, 0, -1)
    total = red.sum()
    if not total:
        return 0.0
    recall = (wide.astype(bool) & red).sum() / total
    if red_wide is None:
        red_wide = cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    drawn = thin.astype(bool)
    precision = (drawn & red_wide).sum() / max(drawn.sum(), 1)
    return float(recall * precision)


def _d(p, a, b):
    return float(np.hypot(*(p[a] - p[b])))


def anatomy_penalty(xy):
    """Log-kara za nieanatomiczny kształt dłoni (współrzędne płótna, X rozciągnięte do proporcji kadru).
    Reguły są celowo luźne – mają odrzucać pomyłki numeracji, a nie nietypowe pozy."""
    p = xy.astype(float).copy()
    p[:, 0] *= ASPECT_X
    pen = 0.0
    meta = [_d(p, 0, m) for m in (5, 9, 13, 17)]
    palm = max(np.median(meta), 1.0)
    # 1) stawy MCP ułożone po kolei wokół nadgarstka (5 → 9 → 13 → 17 w jednym kierunku)
    ang = [math.atan2(p[m, 1] - p[0, 1], p[m, 0] - p[0, 0]) for m in (5, 9, 13, 17)]
    steps = [(ang[i + 1] - ang[i] + math.pi) % (2 * math.pi) - math.pi for i in range(3)]
    if not (all(s > -0.05 for s in steps) or all(s < 0.05 for s in steps)):
        pen += math.log(0.05)
    # 2) odstępy między stawami MCP mniejsze niż długość śródręcza
    spacing = max(_d(p, 5, 9), _d(p, 9, 13), _d(p, 13, 17))
    if spacing > 0.9 * palm:
        pen += math.log(0.05)
    # 3) podstawa kciuka (CMC) blisko nadgarstka
    if _d(p, 0, 1) > 0.9 * palm:
        pen += math.log(0.05)
    # 4) paliczki nie dłuższe niż śródręcze, kolejne paliczki nie dłuższe od poprzednich (z zapasem na skrót perspektywy)
    for mcp in (5, 9, 13, 17):
        prox, mid, dist = _d(p, mcp, mcp + 1), _d(p, mcp + 1, mcp + 2), _d(p, mcp + 2, mcp + 3)
        if prox > 1.25 * palm:
            pen += math.log(0.1)
        if mid > 1.6 * max(prox, 0.25 * palm) or dist > 1.6 * max(mid, 0.2 * palm):
            pen += math.log(0.3)
    return pen


def side_rule(xy):
    """Strona dłoni z podstawy kciuka: CMC kciuka (1) leży po tej samej stronie osi dłoni (nadgarstek → MCP
    środkowego) co MCP wskazującego (5), a po przeciwnej niż MCP małego (17). Podstawa kciuka prawie się nie rusza,
    więc reguła działa także przy kciuku schowanym w pięści (W, M, N…). Zwraca True / False / None (dłoń bokiem).
    Pomiar: 100 % na syntetycznych dłoniach; na prawdziwej próbce zgodna z oceną wzrokową liter."""
    p = xy.astype(float) * [ASPECT_X, 1.0]
    axis = p[9] - p[0]
    length = max(float(np.hypot(*axis)), 1e-6)
    side = [(axis[0] * (p[j, 1] - p[0, 1]) - axis[1] * (p[j, 0] - p[0, 0])) / length for j in (1, 5, 17)]
    palm = float(np.mean(np.hypot(*(p[[5, 9, 13, 17]] - p[0]).T)))
    if abs(side[0]) < 0.05 * palm:
        return None
    return bool(side[0] * side[1] > 0 and side[1] * side[2] < 0)


def chirality_penalty(xy):
    rule = side_rule(xy)
    if rule is None:                                   # dłoń bokiem – słaba wskazówka z położenia kciuka
        return 0.0 if thumb_ratio(xy) <= 1 else THUMB_PENALTY
    return 0.0 if rule else SIDE_PENALTY


def thumb_ratio(xy):
    """< 1: kciuk bliżej wskazującego niż małego palca (poprawna numeracja); odbicie daje > 1."""
    d_index = np.hypot(*(xy[2] - xy[5])) + np.hypot(*(xy[3] - xy[5]))
    d_little = np.hypot(*(xy[2] - xy[17])) + np.hypot(*(xy[3] - xy[17]))
    return float(d_index / max(d_little, 1e-6))


# ---------------------------------------------------------------- główna funkcja

def reconstruct(rgb, template=None):
    green, red = color_masks(rgb)
    points, merged, single = candidates(green)
    out = {'status': 'rejected', 'reason': '', 'n_candidates': len(points), 'n_hidden': -1,
           'min_edge': 0.0, 'explained_red': 0.0, 'margin': 0.0, 'thumb_ratio': 0.0, 'n_merged': int(sum(merged)),
           'xy': None, 'hidden': None}
    if len(points) < 15:
        out['reason'] = 'za_malo_punktow'
        return out
    if len(points) - sum(merged) > 21:
        out['reason'] = 'za_duzo_punktow'
        return out
    s, cov = edge_scores(points, red)
    found = beam_search(s, cov, merged, points, template)
    if not found:
        out['reason'] = 'brak_dopasowania_topologii'
        return out

    pairs = drawing_order_pairs(points, green)
    red_wide = cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    def full_score(assign):
        ex = explained_red(points, assign, red, red_wide)
        total = (local_score(s, assign, points, template) + EXPLAIN_WEIGHT * math.log(max(ex, 1e-3))
                 + anatomy_penalty(points[assign]) + order_penalty(assign, pairs))
        total += chirality_penalty(points[assign])
        return total, ex

    scored = []
    for assign in found:
        total, ex = full_score(assign)
        scored.append((total, assign, ex))
    scored.sort(key=lambda e: -e[0])

    def same(a, b):
        """To samo rozwiązanie, jeśli każdy staw ląduje w tym samym miejscu (± 4 px), także po odbiciu."""
        for cand in (b, b[TWIN]):
            if np.hypot(*(points[a] - points[cand]).T).max() <= 4.0:
                return True
        return False

    # Najlepsze RÓŻNE rozwiązania (dla etapu 2: rozstrzyganie remisów na poziomie litery).
    distinct = []
    for total, assign, ex in scored:
        if not any(same(assign, d[1]) for d in distinct):
            distinct.append((total, assign, ex))
        if len(distinct) == N_ALTERNATIVES:
            break

    def describe(total, assign, ex):
        counts = np.bincount(assign, minlength=len(points))
        hidden = (counts[assign] > 1).astype(int)
        edges = [cov[assign[a], assign[b]] for a, b in CONNECTIONS if assign[a] != assign[b]]
        edges = [e for e in edges if not np.isnan(e)]
        return {'score': float(total), 'xy': points[assign], 'hidden': hidden, 'explained_red': float(ex),
                'min_edge': float(min(edges) if edges else 0), 'n_hidden': int(21 - len(set(assign.tolist())))}

    alternatives = []
    for total, assign, ex in distinct:
        # obie numeracje (odbicie wskazujący ↔ mały); bez szablonu decyduje słaba reguła kciuka
        for variant in (assign, assign[TWIN]):
            total_v, ex_v = full_score(variant)
            d = describe(total_v, variant, ex_v)
            d['thumb_ratio'] = thumb_ratio(d['xy'])
            alternatives.append(d)
    alternatives.sort(key=lambda d: -d['score'])
    out['alternatives'] = alternatives

    best = alternatives[0]
    rival = next((d for d in alternatives[1:]
                  if np.hypot(*(d['xy'] - best['xy']).T).max() > 4.0), None)
    margin = best['score'] - rival['score'] if rival is not None else 99.0
    out.update({'n_hidden': best['n_hidden'], 'min_edge': best['min_edge'], 'explained_red': best['explained_red'],
                'margin': float(margin), 'thumb_ratio': best['thumb_ratio'], 'xy': best['xy'], 'hidden': best['hidden']})
    out['reason'] = quality_problem(best)
    if not out['reason'] and margin < MIN_MARGIN:
        out['reason'] = 'niejednoznaczne'
    if not out['reason']:
        out['status'] = 'ok'
    return out


def quality_problem(d):
    """Twarde kryteria jakości jednego rozwiązania (niezależne od konkurencji)."""
    if d['n_hidden'] > 2:
        return 'za_duzo_ukrytych_stawow'
    if d['min_edge'] < 0.6:
        return 'slabo_widoczna_kosc'
    if d['explained_red'] < 0.85:
        return 'niewyjasnione_kosci'
    return ''


# ---------------------------------------------------------------- przetwarzanie datasetu

FIELDS = (['letter', 'file', 'index', 'status', 'reason', 'n_candidates', 'n_merged', 'n_hidden',
           'min_edge', 'explained_red', 'margin', 'thumb_ratio']
          + [f'{c}{j}' for j in range(N_JOINTS) for c in ('x', 'y')]
          + [f'hidden{j}' for j in range(N_JOINTS)])


def _process(path):
    letter = basename(os.path.dirname(path))
    stem = os.path.splitext(basename(path))[0]
    rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    r = reconstruct(rgb)
    row = {'letter': letter, 'file': basename(path), 'index': int(stem) if stem.isdigit() else -1,
           'status': r['status'], 'reason': r['reason']}
    for k in ('n_candidates', 'n_merged', 'n_hidden'):
        row[k] = r[k]
    for k in ('min_edge', 'explained_red', 'margin', 'thumb_ratio'):
        row[k] = round(r[k], 4)
    for j in range(N_JOINTS):
        row[f'x{j}'] = round(float(r['xy'][j, 0]), 3) if r['xy'] is not None else ''
        row[f'y{j}'] = round(float(r['xy'][j, 1]), 3) if r['xy'] is not None else ''
        row[f'hidden{j}'] = int(r['hidden'][j]) if r['hidden'] is not None else ''
    m = 2 * N_ALTERNATIVES
    alt = {'xy': np.full((m, 21, 2), np.nan, np.float32), 'score': np.full(m, -np.inf, np.float32),
           'hidden': np.zeros((m, 21), np.int8), 'explained_red': np.zeros(m, np.float32),
           'min_edge': np.zeros(m, np.float32), 'n_hidden': np.full(m, 99, np.int8),
           'thumb_ratio': np.zeros(m, np.float32)}
    for k, d in enumerate(r.get('alternatives', [])[:m]):
        for key in alt:
            alt[key][k] = d[key]
    return row, alt


def draw_qa(rgb, row):
    """Podgląd do kontroli: numery stawów na obrazie, kolor = palec."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    colors = [(0, 0, 0), (255, 0, 255), (255, 128, 0), (0, 160, 0), (0, 128, 255), (128, 0, 128)]
    if row['x0'] != '':
        xy = np.array([[row[f'x{j}'], row[f'y{j}']] for j in range(21)], float)
        for j in range(21):
            finger = 0 if j == 0 else (j - 1) // 4 + 1
            p = tuple(np.round(xy[j]).astype(int))
            cv2.putText(img, str(j), (p[0] + 4, p[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[finger], 1)
    cv2.putText(img, f"{row['letter']} {row['file']} {row['status']} {row['reason']}", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True, help='folder z podfolderami A…Z')
    parser.add_argument('--out', default='landmarks.csv')
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--limit', type=int, default=0, help='max obrazów na literę (0 = wszystkie)')
    parser.add_argument('--qa', default='', help='folder na podglądy (losowe przyjęte + odrzucone)')
    args = parser.parse_args()

    files = []
    for folder in sorted(glob(join(args.dataset, '*'))):
        fs = sorted(glob(join(folder, '*.*')), key=lambda f: (len(basename(f)), basename(f)))
        files += fs[:args.limit] if args.limit else fs
    print(f'Obrazów: {len(files)}')
    rows, alts = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for i, (row, alt) in enumerate(pool.map(_process, files, chunksize=16)):
            rows.append(row)
            alts.append(alt)
            if (i + 1) % 1000 == 0:
                print(f'  {i + 1}/{len(files)}', flush=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    # Wszystkie warianty każdego obrazu – wejście etapu 2 (resolve_letters.py).
    alt_path = os.path.splitext(args.out)[0] + '_alternatives.npz'
    np.savez_compressed(alt_path, letter=np.array([r['letter'] for r in rows]),
                        filename=np.array([r['file'] for r in rows]),
                        **{k: np.stack([a[k] for a in alts]) for k in alts[0]})
    print(f'Warianty: {alt_path}')

    ok = sum(r['status'] == 'ok' for r in rows)
    print(f'Przyjęte: {ok}/{len(rows)} ({100 * ok / max(len(rows), 1):.1f}%) → {args.out}')
    letters = sorted({r['letter'] for r in rows})
    for letter in letters:
        lr = [r for r in rows if r['letter'] == letter]
        print(f'  {letter}: {sum(r["status"] == "ok" for r in lr)}/{len(lr)}')
    reasons = {}
    for r in rows:
        if r['status'] != 'ok':
            reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
    print('Powody odrzucenia:', reasons)

    if args.qa:
        os.makedirs(args.qa, exist_ok=True)
        rng = np.random.default_rng(0)
        by_path = {join(args.dataset, r['letter'], r['file']): r for r in rows}
        picks = [p for p, r in by_path.items() if r['status'] == 'ok']
        rejected = [p for p, r in by_path.items() if r['status'] != 'ok']
        for p in list(rng.permutation(picks)[:60]) + list(rng.permutation(rejected)[:40]):
            r = by_path[p]
            rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            cv2.imwrite(join(args.qa, f"{r['status']}_{r['letter']}_{r['file']}.png"), draw_qa(rgb, r))
        print(f'Podglądy: {args.qa}')


if __name__ == '__main__':
    sys.exit(main())
