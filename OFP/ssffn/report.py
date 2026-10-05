"""Seven paper metrics, pooled over modules rather than averaged over folds."""
import math
import pandas as pd

METRICS = ('Precision', 'Recall', 'F1', 'Accuracy', 'AvgLead_hours', 'AFWS', 'FP')


def metrics_from_decisions(detail):
    if detail.empty:
        raise ValueError('Cannot report an empty evaluation')
    if detail.file_name.duplicated().any():
        raise ValueError('A module was evaluated more than once; cannot pool these decisions')
    true = detail.true_label.gt(0)
    hit = detail.hit.gt(0)
    predicted = detail.valid_predict_positive.gt(0)
    tp, fp = int(hit.sum()), int((~true & predicted).sum())
    fn, tn = int(true.sum()) - tp, int((~true & ~predicted).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
    accuracy = (tp + tn) / len(detail)
    lead = float(pd.to_numeric(detail.loc[hit, 'lead_hour']).mean()) if tp else 0.
    return dict(zip(METRICS, (precision, recall, f1, accuracy, lead,
                              f1 + accuracy + math.tanh(lead), fp)))


def markdown_table(rows):
    frame = pd.DataFrame(rows)[['Model', *METRICS]]
    lines = ['| Model | ' + ' | '.join(METRICS) + ' |', '|---|' + '---:|' * len(METRICS)]
    for _, row in frame.iterrows():
        cells = [str(row.Model)]
        for key in METRICS:
            best = frame[key].min() if key == 'FP' else frame[key].max()
            value = str(int(row[key])) if key == 'FP' else f'{row[key]:.3f}'
            cells.append(f'**{value}**' if row[key] == best else value)
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'
