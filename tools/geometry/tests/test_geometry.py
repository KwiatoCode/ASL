"""
Testy części Python (Windows/macOS/Linux):   python -m unittest discover -s tests -v   (z folderu tools/geometry)

Sprawdzają: niezmienniczość cech, zgodność z zapisanymi przykładami dla Swift, zgodność stałych Python ↔ Swift,
poprawność matematyki eksportu drzew do Core ML (na atrapie konstruktora) i rekonstrukcję prostej dłoni.
"""

import json
import os
import re
import sys
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import hand_features as HF  # noqa: E402
import synthetic_hands as S  # noqa: E402

SWIFT = os.path.join(ROOT, '..', '..', 'ASSN', 'Recognition', 'VisionFeatureExtractor.swift')


def hand_points(seed):
    rng = np.random.default_rng(seed)
    while True:
        n = S.project(S.build_hand(S.sample_params(rng)), rng)
        if n is not None:
            return n * [S.CAMERA_W, S.CAMERA_H]


class FeatureTests(unittest.TestCase):
    def test_count_and_sets(self):
        self.assertEqual(len(HF.FEATURE_NAMES), 110)
        self.assertEqual(len(HF.BASIC), 63)
        self.assertEqual(len(set(HF.FEATURE_NAMES)), 110)

    def test_translation_scale_invariance(self):
        for seed in range(10):
            p = hand_points(seed)
            f = HF.extract(p)
            np.testing.assert_allclose(HF.extract(p + [123.4, -56.7]), f, atol=1e-9)
            np.testing.assert_allclose(HF.extract(p * 0.37), f, atol=1e-9)

    def test_canonical_mirror_invariance(self):
        for seed in range(10):
            p = hand_points(seed)
            m = p.copy()
            m[:, 0] = 1000 - m[:, 0]
            np.testing.assert_allclose(HF.extract(m, 'canonical'), HF.extract(p, 'canonical'), atol=1e-9)

    def test_batch_equals_single(self):
        pts = np.stack([hand_points(s) for s in range(5)])
        batch = HF.extract(pts)
        for i in range(5):
            np.testing.assert_allclose(batch[i], HF.extract(pts[i]), atol=1e-12)

    def test_known_values(self):
        # prosta dłoń: palce wyprostowane → zgięcia ≈ 0, prostoliniowość = 1
        p = np.zeros((21, 2))
        for k, x in enumerate([-30, -10, 10, 30]):
            for s in range(4):
                p[5 + 4 * k + s] = [x, -100 - 30 * s]
        for s in range(4):
            p[1 + s] = [-40 - 20 * s, -20 - 20 * s]
        f = dict(zip(HF.FEATURE_NAMES, HF.extract(p)))
        self.assertAlmostEqual(f['C1_bend_index_2'], 0.0, places=9)
        self.assertAlmostEqual(f['B4_extension_middle'], 1.0, places=9)
        # oś dłoni = nadgarstek → MCP środkowego (−10, −100)
        self.assertAlmostEqual(f['D_palm_cos'], -10 / np.hypot(10, 100), places=9)
        self.assertAlmostEqual(f['D_palm_sin'], -100 / np.hypot(10, 100), places=9)


class ParityTests(unittest.TestCase):
    def test_fixtures_match_python(self):
        with open(os.path.join(HERE, 'parity', 'feature_fixtures.json'), encoding='utf-8') as fh:
            data = json.load(fh)
        self.assertEqual(data['version'], HF.FEATURE_VERSION)
        for case in data['cases']:
            got = HF.extract(np.array(case['points']), case['chirality'])
            np.testing.assert_allclose(got, case['features'], atol=1e-12)

    def test_swift_constants(self):
        with open(SWIFT, encoding='utf-8') as fh:
            src = fh.read()
        self.assertIn(f'static let version = "{HF.FEATURE_VERSION}"', src)
        self.assertIn(f'static let featureCount = {len(HF.FEATURE_NAMES)}', src)
        self.assertEqual(len(re.findall(r'f\.append\(', src)), 2 + 1 + 1 + 1 + 1 + 1 + 4 + 5 + 6,
                         'liczba miejsc dopisujących cechy w Swift zmieniła się – sprawdź zgodność z Pythonem')


class ExportTests(unittest.TestCase):
    def test_tree_math_matches_predict_proba(self):
        from sklearn.ensemble import RandomForestClassifier
        rng = np.random.default_rng(0)
        x = rng.normal(size=(600, 12))
        y = np.array(list('ABCD'))[(x[:, 0] > 0) * 2 + (x[:, 1] > 0)]
        model = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0,
                                       class_weight='balanced_subsample').fit(x, y)
        nodes = {}

        class FakeBuilder:
            def __init__(self, features, classes, out):
                self.spec = types.SimpleNamespace(specificationVersion=1)

            def set_default_prediction_value(self, values):
                self.default = values

            def add_leaf_node(self, t, n, values):
                nodes[(t, n)] = ('leaf', values)

            def add_branch_node(self, t, n, f, v, mode, left, right):
                nodes[(t, n)] = ('branch', f, v, left, right)

        ct = types.ModuleType('coremltools')
        ct.models = types.SimpleNamespace(datatypes=types.SimpleNamespace(Array=lambda n: n), MLModel=lambda s: s)
        te = types.ModuleType('coremltools.models.tree_ensemble')
        te.TreeEnsembleClassifier = FakeBuilder
        saved = {k: sys.modules.get(k) for k in ('coremltools', 'coremltools.models', 'coremltools.models.tree_ensemble')}
        sys.modules.update({'coremltools': ct, 'coremltools.models': ct.models, 'coremltools.models.tree_ensemble': te})
        try:
            import export_geometry_coreml as E
            E.build_tree_ensemble(model, x.shape[1])
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        xt = rng.normal(size=(200, 12)).astype(np.float32).astype(np.float64)
        out = np.zeros((len(xt), 4))
        for i, row in enumerate(xt):
            for t in range(20):
                node = nodes[(t, 0)]
                while node[0] == 'branch':
                    node = nodes[(t, node[3] if row[node[1]] <= node[2] else node[4])]
                for c, p in node[1].items():
                    out[i, c] += p
        np.testing.assert_allclose(out, model.predict_proba(xt), atol=1e-12)


class ReconstructionTests(unittest.TestCase):
    def test_open_hand_found_among_alternatives(self):
        import reconstruct_skeleton as R
        rng = np.random.default_rng(11)
        params = S.sample_params(rng)
        for f in params['fingers'].values():
            f['flex'] = np.array([0.05, 0.05, 0.02])
        params.update(abduction=1.1, opposition=0.0, thumb_flex=np.zeros(3), spread=1.3)
        view = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'mirror': False}
        normalized = S.project(S.build_hand(params), rng, view)
        image = S.jpeg(S.draw_like_mediapipe(normalized), 90)
        truth = S.to_pixel(normalized).astype(float)
        r = R.reconstruct(image)
        errors = [np.linalg.norm(a['xy'] - truth, axis=1).max() for a in r['alternatives']]
        self.assertLess(min(errors), 3.0, 'prawdziwe położenia stawów nie trafiły do wariantów rekonstrukcji')


if __name__ == '__main__':
    unittest.main()
