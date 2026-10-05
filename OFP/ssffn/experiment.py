"""Full-budget experiments and module ablations under explicit split protocols."""
import gc
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from ._engine import training as engine
from ._engine.data import file_label_map
from .inference import predict
from .model import VARIANTS, build_model
from .report import METRICS, markdown_table, metrics_from_decisions
from .splits import fixed_manifest, inner_partitions, validate_index

PACKAGE = Path(__file__).resolve().parent
LABELS = dict(full='SSFFN', no_sensor='w/o SMB', no_statistics='w/o SFB',
              no_sit='w/o SIT', no_hss='w/o HSS')
CONTEXT = {}


class DatasetWithNormalization(engine.SelectedWindowStatDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CONTEXT.update(raw_mean=self.cache.mean[self.raw_indices[:12]].copy(),
                       raw_std=self.cache.std[self.raw_indices[:12]].copy())


def engine_args(data_dir, index_path, output_dir, cache_dir, variant, seed, device, epochs=16):
    config = json.loads((PACKAGE/'configs/split3.json').read_text())
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *config['engine_args']]
        args = engine.parse_args()
    finally:
        sys.argv = old_argv
    args.data_dir, args.index_path = Path(data_dir).resolve(), Path(index_path).resolve()
    args.out_root, args.module_cache_dir = Path(output_dir), Path(cache_dir)
    args.module_variant, args.training_seed, args.device = variant, int(seed), device
    args.epochs = int(epochs)
    args.method_label = LABELS[variant]
    args.experiment_id = f'ssffn_split3_{variant}_s{seed}'
    return args


def verify_completed(run_dir, result, epochs):
    checkpoint = torch.load(run_dir/'training_checkpoint.pt', map_location='cpu', weights_only=True)
    if checkpoint['epoch'] != epochs or len(checkpoint['history']) != epochs:
        raise RuntimeError('Training did not finish the declared budget')
    if not result or not result[0]['val_metrics']:
        raise RuntimeError('Validation selection was not completed')
    if not (run_dir/'aligned_encoder.pt').exists():
        raise RuntimeError('Final inference checkpoint is missing')


def run_experiment(data_dir, index_path, out_root, protocol, variants=('full',),
                   training_seed=42, device='cuda', folds=(1,2,3), smoke=False):
    if protocol not in ('legacy_cv', 'fixed_holdout'):
        raise ValueError('Choose legacy_cv or fixed_holdout explicitly')
    if protocol == 'fixed_holdout' and tuple(folds) != (1,2,3):
        raise ValueError('Fixed-holdout selection requires all three inner folds')
    if not variants or any(v not in VARIANTS for v in variants) or len(set(variants)) != len(variants):
        raise ValueError('Choose unique module variants')
    if not folds or any(f not in (1,2,3) for f in folds) or len(set(folds)) != len(folds):
        raise ValueError('Choose unique folds from 1, 2, 3')
    data_dir, index_path, out_root = Path(data_dir).resolve(), Path(index_path).resolve(), Path(out_root).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f'Use a new output directory; existing results are not overwritten: {out_root}')
    index = engine.read_index(index_path)
    validate_index(index)
    missing = [n for n in index.file_name if not (data_dir/n).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} module CSVs missing, e.g. {missing[:3]}')
    out_root.mkdir(parents=True, exist_ok=True)
    epochs = 3 if smoke else 16  # Exercise the HSS refresh after epoch two.
    config = dict(model='SSFFN-split3', protocol=protocol, epochs=epochs,
                  data_seed=42, training_seed=training_seed, variants=list(variants), folds=list(folds),
                  index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                  kind='synthetic_smoke' if smoke else 'full_budget', status='running',
                  AFWS_contains_MinLead=False, timestamp_policy='legacy_float32')
    (out_root/'run.json').write_text(json.dumps(config, indent=2))
    manifest = fixed_manifest(index) if protocol == 'fixed_holdout' else None
    if manifest is not None:
        manifest.to_csv(out_root/'split_manifest.csv', index=False)
    torch.set_num_threads(2)
    engine.RUN_NAME = 'ssffn'
    engine.SelectedWindowStatDataset = DatasetWithNormalization
    engine.TemporalStatAligner = lambda **kw: build_model(normalization=CONTEXT, **kw)
    rows, fold_rows, validation_rows = [], [], []
    for variant in variants:
        args = engine_args(data_dir, index_path, out_root/variant, out_root/'cache',
                           variant, training_seed, device, epochs)
        results = []
        for fold in folds:
            partitions = inner_partitions(manifest, fold) if manifest is not None else None
            result = engine.run_fold(fold, args, partitions=partitions)
            run_dir = args.out_root/'ssffn'/f'fold_{fold}'
            verify_completed(run_dir, result, epochs)
            (run_dir/'completed.json').write_text(json.dumps(result, indent=2))
            results.append((fold, result[0], run_dir))
            validation_detail = pd.read_csv(run_dir/'validation/encoder_head_linear/best_module_decisions.csv')
            validation_rows.append(dict(Model=LABELS[variant], Variant=variant, Fold=fold,
                                        Threshold=result[0]['threshold'],
                                        **metrics_from_decisions(validation_detail)))
            pd.DataFrame(validation_rows).to_csv(out_root/'validation_results.csv', index=False)
            if protocol == 'legacy_cv':
                detail = pd.read_csv(run_dir/'evaluation/encoder_head_linear/module_decisions.csv')
                fold_rows.append(dict(Model=LABELS[variant], Variant=variant, Fold=fold,
                                      TrainingSeed=training_seed, **metrics_from_decisions(detail)))
        if protocol == 'fixed_holdout':
            # Select a trained fold model only by its own validation F1; ties
            # use the lower fold number. Do not pool repeated test predictions.
            fold, selected, run_dir = max(results, key=lambda item: (item[1]['val_metrics']['f1_score'], -item[0]))
            selection = dict(fold=fold, criterion='validation F1; lower fold breaks ties',
                             threshold=selected['threshold'], validation_metrics=selected['val_metrics'],
                             checkpoint=str((run_dir/'aligned_encoder.pt').relative_to(out_root)))
            (args.out_root/'selection.json').write_text(json.dumps(selection, indent=2))
            test_names = manifest.loc[manifest.subset.eq('test'), 'file_name'].tolist()
            pred_dir = args.out_root/'fixed_test/predictions'
            predict(run_dir/'aligned_encoder.pt', data_dir, test_names, pred_dir,
                    selected['threshold'], device)
            _, detail = engine.evaluate_prediction_output(pred_dir, data_dir, args.out_root/'fixed_test/evaluation', 0.)
        else:
            detail = pd.concat([pd.read_csv(path/'evaluation/encoder_head_linear/module_decisions.csv')
                                for _, _, path in results], ignore_index=True)
        row = dict(Model=LABELS[variant], Variant=variant, TrainingSeed=training_seed,
                   Protocol=protocol, **metrics_from_decisions(detail))
        rows.append(row)
        detail.to_csv(args.out_root/'module_decisions.csv', index=False)
        pd.DataFrame(rows).to_csv(out_root/'results.csv', index=False)
        (out_root/'RESULTS.md').write_text(f'Protocol: `{protocol}`. Kind: `{config["kind"]}`.\n\n'+markdown_table(rows), encoding='utf-8')
        if fold_rows:
            pd.DataFrame(fold_rows).to_csv(out_root/'per_fold_results.csv', index=False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    config['status'] = 'complete'
    config['reported_modules'] = len(detail)
    (out_root/'run.json').write_text(json.dumps(config, indent=2))
    print(markdown_table(rows))
    return rows
