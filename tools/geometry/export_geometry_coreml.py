"""
Eksport Random Forest (cechy geometryczne) do Core ML + weryfikacja zgodności z scikit-learn – bez Maca.

  python export_geometry_coreml.py --model wyniki_rf/model_rf.joblib --check wyniki_rf/model_rf_ocena.joblib \\
         --output ..\\..\\ASSN\\ASLGeometryRF.mlmodel

Model Core ML:
  wejście  `features`  MultiArray(Double, n)  – cechy z VisionFeatureExtractor w kolejności z metadanych
  wyjście  `classLabel` (A…Z) + `classProbability` {litera: p}
  metadane feature_version, feature_set, chirality, indices – aplikacja odmawia użycia modelu o innej wersji cech.

Weryfikacja: drzewa są odczytywane z zapisanej specyfikacji Core ML (protobuf) i przeliczane w Pythonie
na zbiorze testowym; porównanie z `predict_proba` scikit-learn (zgodność liter i największa różnica p).
To sprawdza konwersję bez uruchamiania Core ML (który działa tylko na macOS/iOS).

Zależności: pip install scikit-learn coremltools joblib numpy   (dowolna wersja scikit-learn – drzewa są
przepisywane bezpośrednio, bez konwertera sklearn z coremltools)
"""

import argparse
import os

import joblib
import numpy as np

from hand_features import FEATURE_NAMES, FEATURE_SETS, FEATURE_VERSION

MODEL_NAME = 'ASLGeometryRF'


def convert(bundle):
    import coremltools as ct

    model, fs, ch = bundle['model'], bundle['feature_set'], bundle['chirality']
    if bundle.get('feature_version') != FEATURE_VERSION:
        raise SystemExit(f"Model trenowano na cechach {bundle.get('feature_version')}, a kod ma {FEATURE_VERSION}.")
    n = len(FEATURE_SETS[fs])
    mlmodel = build_tree_ensemble(model, n)
    indices = [FEATURE_NAMES.index(name) for name in FEATURE_SETS[fs]]
    mlmodel.short_description = (f'ASL A–Z z geometrii dłoni (Apple Vision → {n} cech {FEATURE_VERSION} → '
                                 f'Random Forest, {model.n_estimators} drzew)')
    mlmodel.input_description['features'] = f'Cechy VisionFeatureExtractor ({fs}, chirality={ch})'
    meta = mlmodel.user_defined_metadata
    meta['feature_version'] = FEATURE_VERSION
    meta['feature_set'] = fs
    meta['chirality'] = ch
    meta['n_features'] = str(n)
    meta['indices'] = ','.join(map(str, indices))
    meta['classes'] = ','.join(map(str, model.classes_))
    return mlmodel


def build_tree_ensemble(model, n_features):
    """Zespół drzew Core ML zbudowany wprost z drzew scikit-learn (bez konwertera sklearn z coremltools,
    który obsługuje tylko wybrane wersje scikit-learn). Liść = rozkład klas znormalizowany do 1 i podzielony
    przez liczbę drzew – dokładnie tak liczy RandomForestClassifier.predict_proba."""
    import coremltools as ct
    from coremltools.models.tree_ensemble import TreeEnsembleClassifier

    classes = [str(c) for c in model.classes_]
    builder = TreeEnsembleClassifier([('features', ct.models.datatypes.Array(n_features))], classes, 'classLabel')
    # Wartość startowa = 0 dla każdej klasy; ustawia też liczbę wymiarów wyniku (numPredictionDimensions = liczba klas).
    builder.set_default_prediction_value([0.0] * len(classes))
    n_trees = len(model.estimators_)
    for tree_id, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        for node in range(tree.node_count):
            left, right = int(tree.children_left[node]), int(tree.children_right[node])
            if left == -1:
                value = tree.value[node][0].astype(float)
                value = value / value.sum()
                builder.add_leaf_node(tree_id, node, {int(c): float(p) / n_trees for c, p in enumerate(value) if p > 0})
            else:
                # scikit-learn: lewe dziecko, gdy x ≤ próg
                builder.add_branch_node(tree_id, node, int(tree.feature[node]), float(tree.threshold[node]),
                                        'BranchOnValueLessThanEqual', left, right)
    spec = builder.spec
    spec.specificationVersion = max(spec.specificationVersion, 1)
    return ct.models.MLModel(spec)


def evaluate_spec(spec, x):
    """Przeliczenie zespołu drzew zapisanego w specyfikacji Core ML (TreeEnsembleClassifier)."""
    if spec.WhichOneof('Type') != 'treeEnsembleClassifier':
        raise SystemExit(f"Nieoczekiwany typ modelu Core ML: {spec.WhichOneof('Type')}")
    clf = spec.treeEnsembleClassifier
    ens = clf.treeEnsemble
    tree_node = type(ens.nodes[0])
    behavior = {name: tree_node.TreeNodeBehavior.Value(name) for name in tree_node.TreeNodeBehavior.keys()}
    nodes = {}
    for node in ens.nodes:
        nodes[(node.treeId, node.nodeId)] = node
    roots = sorted({t for t, _ in nodes})
    labels = list(clf.stringClassLabels.vector) or [str(v) for v in clf.int64ClassLabels.vector]
    dims = ens.numPredictionDimensions
    if dims != len(labels) or len(ens.basePredictionValue) != len(labels):
        raise SystemExit(f'Błędna specyfikacja Core ML: numPredictionDimensions={dims}, '
                         f'basePredictionValue={len(ens.basePredictionValue)}, klas={len(labels)}')
    base = np.array(list(ens.basePredictionValue), float)
    out = np.tile(base, (len(x), 1))
    compare = {
        behavior['BranchOnValueLessThanEqual']: lambda v, t: v <= t,
        behavior['BranchOnValueLessThan']: lambda v, t: v < t,
        behavior['BranchOnValueGreaterThanEqual']: lambda v, t: v >= t,
        behavior['BranchOnValueGreaterThan']: lambda v, t: v > t,
        behavior['BranchOnValueEqual']: lambda v, t: v == t,
        behavior['BranchOnValueNotEqual']: lambda v, t: v != t,
    }
    leaf = behavior['LeafNode']
    for i, row in enumerate(x):
        for tree in roots:
            node = nodes[(tree, 0)]
            while node.nodeBehavior != leaf:
                go_true = compare[node.nodeBehavior](row[node.branchFeatureIndex], node.branchFeatureValue)
                node = nodes[(tree, node.trueChildNodeId if go_true else node.falseChildNodeId)]
            for ev in node.evaluationInfo:
                out[i, ev.evaluationIndex] += ev.evaluationValue
    transform = clf.postEvaluationTransform
    names = type(clf).TreeEnsemblePostEvaluationTransform if hasattr(type(clf), 'TreeEnsemblePostEvaluationTransform') else None
    if names is not None and transform != names.Value('NoTransform'):
        if transform == names.Value('Classification_SoftMax'):
            e = np.exp(out - out.max(1, keepdims=True))
            out = e / e.sum(1, keepdims=True)
        else:
            raise SystemExit(f'Nieobsłużona transformacja wyniku: {transform}')
    return labels, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='model_rf.joblib (model do aplikacji)')
    parser.add_argument('--check', help='model_rf_ocena.joblib (model oceniany + zbiór testowy) – weryfikacja')
    parser.add_argument('--output', default=os.path.join('..', '..', 'ASSN', f'{MODEL_NAME}.mlmodel'))
    parser.add_argument('--check-samples', type=int, default=500)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    mlmodel = convert(bundle)
    mlmodel.save(args.output)
    print(f'Zapisano {os.path.abspath(args.output)} ({os.path.getsize(args.output) / 1e6:.1f} MB)')

    if args.check:
        check = joblib.load(args.check)
        # Aplikacja (i scikit-learn wewnętrznie) używa cech zaokrąglonych do float32.
        x_test = check['x_test'][:args.check_samples].astype(np.float32).astype(np.float64)
        conv = convert(check)
        labels, probs = evaluate_spec(conv.get_spec(), x_test)
        ref = check['model'].predict_proba(x_test)
        ref_labels = [str(c) for c in check['model'].classes_]
        order = [labels.index(c) for c in ref_labels]
        probs = probs[:, order]
        agree = (probs.argmax(1) == ref.argmax(1)).mean()
        print(f'Weryfikacja na {len(x_test)} próbkach testowych: zgodność liter {100 * agree:.2f}%, '
              f'największa różnica prawdopodobieństwa {np.abs(probs - ref).max():.2e}')
        if agree < 1.0 or np.abs(probs - ref).max() > 1e-4:
            raise SystemExit('UWAGA: Core ML i scikit-learn dają różne wyniki – nie używaj tego modelu.')


if __name__ == '__main__':
    main()
