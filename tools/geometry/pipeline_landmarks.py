"""
Pełna rekonstrukcja punktów dłoni dla datasetu: etap 1 → etap 2 (EM) → szablony liter → etap 1 z szablonem
→ etap 2 → wynik. Każdy przebieg jest zapisywany, więc przerwany proces można wznowić.

  python pipeline_landmarks.py --dataset <folder A…Z> --out wyniki_rekonstrukcji --jobs 6

Wynik (folder --out):
  landmarks.csv          przyjęte i odrzucone obrazy, 21 punktów (piksele płótna 400×400), flagi stawów ukrytych
  przebieg_*.npz         warianty każdego obrazu z każdego przebiegu
  szablony_*.npz         szablony kształtu liter
  raport_rekonstrukcji.md  odsetek przyjętych (ogółem, per litera), powody odrzuceń
  qa/                    podglądy: losowe przyjęte i odrzucone obrazy z numerami stawów
"""

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from os.path import basename, join

import cv2
import numpy as np

import reconstruct_skeleton as R
from resolve_letters import resolve, templates

PASSES = 2       # przebiegi z szablonem po przebiegu bez szablonu


def _alternatives(job):
    item, template = job
    image = cv2.cvtColor(cv2.imread(item), cv2.COLOR_BGR2RGB) if isinstance(item, str) else item
    r = R.reconstruct(image, template)
    m = 2 * R.N_ALTERNATIVES
    alt = {'xy': np.full((m, 21, 2), np.nan, np.float32), 'score': np.full(m, -np.inf, np.float32),
           'hidden': np.zeros((m, 21), np.int8), 'explained_red': np.zeros(m, np.float32),
           'min_edge': np.zeros(m, np.float32), 'n_hidden': np.full(m, 99, np.int8),
           'thumb_ratio': np.zeros(m, np.float32)}
    for k, d in enumerate(r.get('alternatives', [])[:m]):
        for key in alt:
            alt[key][k] = d[key]
    alt['stage1_reason'] = r['reason']
    return alt


def run_pass(items, letters, tmpl, jobs):
    jobs_list = [(it, tmpl.get(str(L)) if tmpl else None) for it, L in zip(items, letters)]
    with ProcessPoolExecutor(jobs) as pool:
        alts = []
        for i, a in enumerate(pool.map(_alternatives, jobs_list, chunksize=8)):
            alts.append(a)
            if (i + 1) % 1000 == 0:
                print(f'    {i + 1}/{len(items)}', flush=True)
    out = {k: np.stack([a[k] for a in alts]) for k in alts[0] if k != 'stage1_reason'}
    out['stage1_reason'] = np.array([a['stage1_reason'] for a in alts])
    out['letter'] = np.asarray(letters)
    return out


def full_pipeline(items, letters, jobs, cache_dir=None, verbose=True):
    def cached(name, fn):
        path = join(cache_dir, name) if cache_dir else None
        if path and os.path.exists(path):
            data = dict(np.load(path, allow_pickle=False))
            if verbose:
                print(f'  wczytano {name}')
            return data
        data = fn()
        if path:
            np.savez_compressed(path, **data)
        return data

    if verbose:
        print('Przebieg 0 (bez szablonu)')
    alt = cached('przebieg_0.npz', lambda: run_pass(items, letters, None, jobs))
    chosen, status, reason, margin, sdist = resolve(alt, verbose=False)
    for p in range(1, PASSES + 1):
        good = np.array([chosen[i] >= 0 and alt['n_hidden'][i, chosen[i]] <= 2 for i in range(len(items))])
        tmpl = templates(alt, chosen, good)
        redo = np.nonzero(status != 'ok')[0]       # przyjęte obrazy zachowują swoje warianty
        if verbose:
            print(f'Przebieg {p} (z szablonami {len(tmpl)} liter); przyjęte po poprzednim: '
                  f'{int((status == "ok").sum())}, do ponownej analizy: {len(redo)}')

        def merged_pass(previous=alt, redo=redo, tmpl=tmpl):
            part = run_pass([items[i] for i in redo], [letters[i] for i in redo], tmpl, jobs)
            out = {k: v.copy() for k, v in previous.items()}
            for k in out:
                if k in part:
                    out[k][redo] = part[k]
            return out
        alt = cached(f'przebieg_{p}.npz', merged_pass)
        chosen, status, reason, margin, sdist = resolve(alt, verbose=False)
    return alt, chosen, status, reason, margin, sdist


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', default='wyniki_rekonstrukcji')
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--limit', type=int, default=0, help='max obrazów na literę (0 = wszystkie)')
    parser.add_argument('--keep-unverified', action='store_true',
                        help='nie odrzucaj liter z nierozstrzygniętą stroną dłoni (tylko po sprawdzeniu qa/)')
    args = parser.parse_args()
    os.makedirs(join(args.out, 'qa'), exist_ok=True)

    items, letters, names = [], [], []
    for folder in sorted(glob(join(args.dataset, '*'))):
        if not os.path.isdir(folder):
            continue
        fs = sorted(glob(join(folder, '*.*')), key=lambda f: (len(basename(f)), basename(f)))
        fs = fs[:args.limit] if args.limit else fs
        items += fs
        letters += [basename(folder)] * len(fs)
        names += [basename(f) for f in fs]
    print(f'Obrazów: {len(items)}, liter: {len(set(letters))}')

    alt, chosen, status, reason, margin, sdist = full_pipeline(items, letters, args.jobs, args.out)
    ok = status == 'ok'
    # Strona dłoni (reguła podstawy kciuka): próbka z kciukiem po stronie małego palca ma odbitą numerację
    # → odrzucona. Litera, w której strony dłoni w większości nie da się rozstrzygnąć (dłoń bokiem), jest
    # odrzucana w całości – nie ma jak sprawdzić jej numeracji. --keep-unverified wyłącza oba filtry.
    flagged = []
    for L in sorted(set(letters)):
        idx = np.nonzero((np.array(letters) == L) & ok)[0]
        if not idx.size or args.keep_unverified:
            continue
        rules = [R.side_rule(alt['xy'][i, chosen[i]]) for i in idx]
        decided = [r for r in rules if r is not None]
        if len(decided) < 0.5 * len(rules):
            flagged.append(L)
            status[idx] = 'odrzucone'
            reason[idx] = 'strona_dloni_niepewna_dla_litery'
            continue
        for i, rule in zip(idx, rules):
            if rule is not None and not rule:
                status[i] = 'odrzucone'
                reason[i] = 'odbita_numeracja'
    ok = status == 'ok'
    if flagged:
        print(f'Litery odrzucone w całości (dłoń bokiem – strony dłoni nie da się sprawdzić): {", ".join(flagged)}')

    fields = (['letter', 'file', 'index', 'status', 'reason', 'final_margin', 'shape_dist_sigma', 'n_hidden',
               'explained_red', 'min_edge']
              + [f'{c}{j}' for j in range(21) for c in ('x', 'y')] + [f'hidden{j}' for j in range(21)])
    with open(join(args.out, 'landmarks.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(items)):
            k = chosen[i]
            stem = os.path.splitext(names[i])[0]
            row = {'letter': letters[i], 'file': names[i], 'index': int(stem) if stem.isdigit() else -1,
                   'status': status[i], 'reason': reason[i], 'final_margin': round(float(margin[i]), 3),
                   'shape_dist_sigma': round(float(sdist[i]), 3)}
            if ok[i]:
                row.update({'n_hidden': int(alt['n_hidden'][i, k]),
                            'explained_red': round(float(alt['explained_red'][i, k]), 4),
                            'min_edge': round(float(alt['min_edge'][i, k]), 4)})
                for j in range(21):
                    row[f'x{j}'] = round(float(alt['xy'][i, k, j, 0]), 3)
                    row[f'y{j}'] = round(float(alt['xy'][i, k, j, 1]), 3)
                    row[f'hidden{j}'] = int(alt['hidden'][i, k, j])
            writer.writerow(row)

    lines = ['# Rekonstrukcja punktów dłoni – raport', '',
             f'- obrazów: {len(items)}', f'- przyjętych: {int(ok.sum())} ({100 * ok.mean():.1f}%)', '',
             'Kolumna „przyjęte” = przed filtrem strony dłoni; „%” = ostatecznie przyjęte do treningu.', '',
             'Kontrola strony dłoni: odsetek przyjętych próbek, w których podstawa kciuka leży po stronie palca',
             'wskazującego (reguła anatomiczna), oraz odsetek nierozstrzygniętych (dłoń bokiem). Litera z niskim',
             'odsetkiem lub w większości nierozstrzygnięta może mieć odbitą numerację – sprawdź ją w qa/.', '',
             '| litera | przyjęte | wszystkie | % | strona dłoni zgodna | nierozstrzygnięte | uwaga |',
             '|---|---|---|---|---|---|---|']
    for L in sorted(set(letters)):
        sel = np.array(letters) == L
        idx = np.nonzero(sel & (ok | np.isin(reason, ['strona_dloni_niepewna_dla_litery', 'odbita_numeracja'])))[0]
        rules = [R.side_rule(alt['xy'][i, chosen[i]]) for i in idx]
        decided = [r for r in rules if r is not None]
        agree = np.mean(decided) if decided else float('nan')
        undecided = 1 - len(decided) / max(len(rules), 1)
        notes = []
        if decided and agree < 0.9:
            notes.append('odbite próbki odrzucone')
        if idx.size and undecided > 0.5:
            notes.append('dłoń bokiem – litera odrzucona')
        if len(idx) < 0.25 * sel.sum():
            notes.append('mało przyjętych')
        agree_txt = f'{100 * agree:.0f}%' if decided else '–'
        lines.append(f'| {L} | {len(idx)} | {int(sel.sum())} | {100 * ok[sel].mean():.1f} | '
                     f'{agree_txt} | {100 * undecided:.0f}% | {", ".join(notes)} |')
    vals, counts = np.unique(reason[~ok], return_counts=True)
    lines += ['', '## Powody odrzucenia', '', '| powód | liczba |', '|---|---|']
    lines += [f'| {v} | {c} |' for v, c in zip(vals, counts)]
    open(join(args.out, 'raport_rekonstrukcji.md'), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    rng = np.random.default_rng(0)
    for i in list(rng.permutation(np.nonzero(ok)[0])[:80]) + list(rng.permutation(np.nonzero(~ok)[0])[:40]):
        rgb = cv2.cvtColor(cv2.imread(items[i]), cv2.COLOR_BGR2RGB)
        xy = alt['xy'][i, chosen[i]] if ok[i] else None
        cv2.imwrite(join(args.out, 'qa', f'{status[i]}_{letters[i]}_{names[i]}.png'),
                    draw_qa(rgb, xy, f'{letters[i]} {names[i]} {status[i]} {reason[i]}'))


FINGER_COLORS = [(0, 0, 0), (200, 0, 200), (255, 90, 0), (0, 160, 0), (0, 140, 255), (0, 0, 150)]   # BGR


def draw_qa(rgb, xy, title):
    """Podgląd do kontroli wzrokowej: kości w kolorach palców (kciuk fioletowy, wskazujący niebieski,
    środkowy zielony, serdeczny pomarańczowy, mały ciemnoczerwony) i numery stawów, 2× powiększone."""
    img = cv2.cvtColor((rgb * 0.35 + 255 * 0.65).astype(np.uint8), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    if xy is not None:
        p = np.round(np.asarray(xy) * 2).astype(int)
        for a, b in R.CONNECTIONS:
            finger = 0 if b in (5, 9, 13, 17) and a in (0, 5, 9, 13) else (b - 1) // 4 + 1
            cv2.line(img, tuple(map(int, p[a])), tuple(map(int, p[b])), FINGER_COLORS[finger], 2)
        for j in range(21):
            finger = 0 if j == 0 else (j - 1) // 4 + 1
            cv2.putText(img, str(j), (int(p[j, 0]) + 5, int(p[j, 1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        FINGER_COLORS[finger], 1)
    cv2.putText(img, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


if __name__ == '__main__':
    sys.exit(main())
