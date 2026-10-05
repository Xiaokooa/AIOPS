"""Run with python -m OFP.ssffn from the repository root."""
import argparse
import json
from pathlib import Path

from .model import VARIANTS


def main():
    parser = argparse.ArgumentParser(description='SSFFN paper model, module ablations and first-warning evaluation')
    commands = parser.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='16 epochs per fold; validation-only threshold selection')
    train.add_argument('--data-dir', type=Path, required=True)
    train.add_argument('--index', type=Path, required=True)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--variant', choices=VARIANTS, default='full')
    train.add_argument('--all-ablations', action='store_true')
    train.add_argument('--training-seed', type=int, default=42)
    train.add_argument('--device', default='cuda')
    split = commands.add_parser('split', help='Generate the fixed 80:20 stratified split, seed 42')
    split.add_argument('--index', type=Path, required=True)
    split.add_argument('--output', type=Path, required=True)
    pred = commands.add_parser('predict', help='Score module CSVs with a trained checkpoint')
    pred.add_argument('--checkpoint', type=Path, required=True)
    pred.add_argument('--data-dir', type=Path, required=True)
    pred.add_argument('--output', type=Path, required=True)
    pred.add_argument('--threshold', type=float, required=True, help='Use the threshold selected on validation data')
    pred.add_argument('--device', default='cpu')
    evaluate = commands.add_parser('evaluate', help='Report all seven metrics from prediction CSVs')
    evaluate.add_argument('--predictions', type=Path, required=True)
    evaluate.add_argument('--labels', type=Path, required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    smoke = commands.add_parser('smoke', help='Synthetic-data integration check; not a paper experiment')
    smoke.add_argument('--output', type=Path, required=True)
    smoke.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.command == 'train':
        from .experiment import run_experiment
        run_experiment(args.data_dir, args.index, args.output,
                       VARIANTS if args.all_ablations else (args.variant,),
                       args.training_seed, args.device)
    elif args.command == 'split':
        from .splits import fixed_manifest
        import pandas as pd
        frame = fixed_manifest(pd.read_csv(args.index))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(args.output)
        frame.to_csv(args.output, index=False)
        print(frame.groupby(['subset', 'Label']).size().to_string())
    elif args.command == 'predict':
        from .inference import predict
        predict(args.checkpoint, args.data_dir, sorted(p.name for p in args.data_dir.glob('*.csv')),
                args.output, args.threshold, args.device)
    elif args.command == 'evaluate':
        from ._engine.evaluation import evaluate_prediction_output
        from .report import metrics_from_decisions
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(args.output)
        files = list(args.predictions.glob('*.csv'))
        if not files or any(not (args.labels/f.name).is_file() for f in files):
            raise ValueError('Every prediction CSV must have a matching label CSV')
        _, detail = evaluate_prediction_output(args.predictions, args.labels, args.output, 0.)
        report = metrics_from_decisions(detail)
        (args.output/'paper_metrics.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    else:
        from .smoke import run_smoke
        run_smoke(args.output, args.device)


if __name__ == '__main__':
    main()
