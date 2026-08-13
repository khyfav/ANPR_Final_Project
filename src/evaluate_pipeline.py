import argparse
from collections import Counter
from pathlib import Path
import pandas as pd
from pipeline import load_model, read_plate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--image-dir', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--model', required=True)
    args = p.parse_args()

    labels = pd.read_csv(args.labels)
    model, device = load_model(args.model)

    total_plates = correct_plates = 0
    seg_success = 0
    char_total = char_correct = 0
    confusions = Counter()
    rows = []

    for r in labels.itertuples(index=False):
        truth = str(r.plate_text)
        result = read_plate(Path(args.image_dir) / r.filename, model, device)
        pred = result['text']

        total_plates += 1

        # Segmentation is considered successful when the predicted
        # character count matches the known ground-truth plate length.
        segmentation_ok = len(pred) == len(truth)

        plate_char_total = 0
        plate_char_correct = 0

        if segmentation_ok:
            seg_success += 1

            for t, pch in zip(truth, pred):
                char_total += 1
                plate_char_total += 1

                if t == pch:
                    char_correct += 1
                    plate_char_correct += 1
                else:
                    confusions[(t, pch)] += 1

        if pred == truth:
            correct_plates += 1

        # Preserve compatibility with the original simple dataset,
        # which may not contain a state column.
        state = getattr(r, 'state', 'baseline')

        rows.append({
            'filename': r.filename,
            'state': state,
            'condition': r.condition,
            'truth': truth,
            'prediction': pred,
            'plate_confidence': result['plate_confidence'],
            'decision': result['decision'],
            'segmentation_ok': segmentation_ok,
            'plate_correct': pred == truth,
            'char_total': plate_char_total,
            'char_correct': plate_char_correct,
        })

    detail = pd.DataFrame(rows)

    print(f'Segmentation success: {seg_success/total_plates:.3%}')
    print(
        'Character accuracy (on correctly segmented plates): '
        f'{char_correct/max(char_total, 1):.3%}'
    )
    print(f'Plate exact-match accuracy: {correct_plates/total_plates:.3%}')

    print('\nBy condition:')
    by_condition = detail.groupby('condition').agg(
        plates=('filename', 'count'),
        segmentation_rate=('segmentation_ok', 'mean'),
        plate_accuracy=('plate_correct', 'mean'),
        mean_confidence=('plate_confidence', 'mean'),
        char_correct=('char_correct', 'sum'),
        char_total=('char_total', 'sum'),
    )

    by_condition['character_accuracy'] = (
        by_condition['char_correct'] /
        by_condition['char_total'].replace(0, pd.NA)
    )

    print(
        by_condition[
            [
                'plates',
                'segmentation_rate',
                'character_accuracy',
                'plate_accuracy',
                'mean_confidence'
            ]
        ].round(3)
    )

    print('\nBy state:')
    by_state = detail.groupby('state').agg(
        plates=('filename', 'count'),
        segmentation_rate=('segmentation_ok', 'mean'),
        plate_accuracy=('plate_correct', 'mean'),
        mean_confidence=('plate_confidence', 'mean'),
        char_correct=('char_correct', 'sum'),
        char_total=('char_total', 'sum'),
    )

    by_state['character_accuracy'] = (
        by_state['char_correct'] /
        by_state['char_total'].replace(0, pd.NA)
    )

    print(
        by_state[
            [
                'plates',
                'segmentation_rate',
                'character_accuracy',
                'plate_accuracy',
                'mean_confidence'
            ]
        ].round(3)
    )

    print('\nTop confusion pairs:')
    for (t, pch), n in confusions.most_common(10):
        print(f'{t} -> {pch}: {n}')

    detail.to_csv('results_detail.csv', index=False)
    by_condition.to_csv('results_by_condition.csv')
    by_state.to_csv('results_by_state.csv')

    print('\nSaved results_detail.csv')
    print('Saved results_by_condition.csv')
    print('Saved results_by_state.csv')


if __name__ == '__main__':
    main()
