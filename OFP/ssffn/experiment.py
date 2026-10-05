"""Training, validation selection and component ablations for SSFFN."""
import gc
import hashlib
import json
from argparse import Namespace
from pathlib import Path
import pandas as pd
import torch
from ._engine import training as engine
from .inference import predict
from .model import VARIANTS, build_model
from .report import markdown_table, metrics_from_decisions
from .splits import fixed_manifest, inner_partitions, validate_index

PACKAGE=Path(__file__).resolve().parent
LABELS=dict(full='SSFFN',no_sensor='w/o SMB',no_statistics='w/o SFB',no_sit='w/o SIT',no_hss='w/o HSS')

def engine_args(data_dir,index_path,output_dir,cache_dir,variant,seed,device,epochs=16):
    args=Namespace(**json.loads((PACKAGE/'configs/ssffn.json').read_text()))
    args.data_dir,args.index_path=Path(data_dir).resolve(),Path(index_path).resolve()
    args.out_root,args.module_cache_dir=Path(output_dir),Path(cache_dir)
    args.module_variant,args.training_seed,args.device=variant,int(seed),device
    args.epochs=int(epochs)
    args.method_label=LABELS[variant]
    args.experiment_id=f'ssffn_{variant}_s{seed}'
    return args

def verify_completed(run_dir,result,epochs):
    checkpoint=torch.load(run_dir/'training_checkpoint.pt',map_location='cpu',weights_only=True)
    if checkpoint['epoch']!=epochs or len(checkpoint['history'])!=epochs:
        raise RuntimeError('Training did not finish the declared budget')
    if not result or not result[0]['val_metrics']:
        raise RuntimeError('Validation selection was not completed')
    if not (run_dir/'aligned_encoder.pt').exists():
        raise RuntimeError('Final inference checkpoint is missing')

def run_experiment(data_dir,index_path,out_root,variants=('full',),training_seed=42,device='cuda',smoke=False):
    if not variants or any(v not in VARIANTS for v in variants) or len(set(variants))!=len(variants):
        raise ValueError('Choose unique component variants')
    data_dir,index_path,out_root=Path(data_dir).resolve(),Path(index_path).resolve(),Path(out_root).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f'Use a new output directory: {out_root}')
    index=engine.read_index(index_path)
    validate_index(index)
    missing=[n for n in index.file_name if not (data_dir/n).is_file()]
    if missing: raise FileNotFoundError(f'{len(missing)} module CSVs missing, e.g. {missing[:3]}')
    out_root.mkdir(parents=True,exist_ok=True)
    epochs=3 if smoke else 16
    config=dict(model='SSFFN',epochs=epochs,data_seed=42,training_seed=training_seed,
        variants=list(variants),index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        kind='synthetic_smoke' if smoke else 'full_budget',status='running')
    (out_root/'run.json').write_text(json.dumps(config,indent=2))
    manifest=fixed_manifest(index)
    manifest.to_csv(out_root/'split_manifest.csv',index=False)
    torch.set_num_threads(2)
    rows,validation_rows=[],[]
    for variant in variants:
        args=engine_args(data_dir,index_path,out_root/variant,out_root/'cache',variant,training_seed,device,epochs)
        results=[]
        for fold in (1,2,3):
            result=engine.run_fold(fold,args,partitions=inner_partitions(manifest,fold))
            run_dir=args.out_root/'ssffn'/f'fold_{fold}'
            verify_completed(run_dir,result,epochs)
            (run_dir/'completed.json').write_text(json.dumps(result,indent=2))
            results.append((fold,result[0],run_dir))
            detail=pd.read_csv(run_dir/'validation/encoder_head_linear/best_module_decisions.csv')
            validation_rows.append(dict(Model=LABELS[variant],Variant=variant,Fold=fold,
                Threshold=result[0]['threshold'],**metrics_from_decisions(detail)))
            pd.DataFrame(validation_rows).to_csv(out_root/'validation_results.csv',index=False)
        fold,selected,run_dir=max(results,key=lambda item:(item[1]['val_metrics']['f1_score'],-item[0]))
        selection=dict(fold=fold,threshold=selected['threshold'],validation_metrics=selected['val_metrics'],
            checkpoint=str((run_dir/'aligned_encoder.pt').relative_to(out_root)))
        (args.out_root/'selection.json').write_text(json.dumps(selection,indent=2))
        test_names=manifest.loc[manifest.subset.eq('test'),'file_name'].tolist()
        pred_dir=args.out_root/'test/predictions'
        predict(run_dir/'aligned_encoder.pt',data_dir,test_names,pred_dir,selected['threshold'],device)
        _,detail=engine.evaluate_prediction_output(pred_dir,data_dir,args.out_root/'test/evaluation',0.)
        rows.append(dict(Model=LABELS[variant],Variant=variant,TrainingSeed=training_seed,**metrics_from_decisions(detail)))
        detail.to_csv(args.out_root/'module_decisions.csv',index=False)
        pd.DataFrame(rows).to_csv(out_root/'results.csv',index=False)
        (out_root/'RESULTS.md').write_text(f'Run: `{config["kind"]}`.\n\n'+markdown_table(rows),encoding='utf-8')
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    config.update(status='complete',reported_modules=len(detail))
    (out_root/'run.json').write_text(json.dumps(config,indent=2))
    print(markdown_table(rows))
    return rows
