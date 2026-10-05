"""Small, obviously synthetic data for installation and integration checking."""
from pathlib import Path
import numpy as np
import pandas as pd


def generate_data(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data_dir = root/'data'
    data_dir.mkdir()
    rng = np.random.default_rng(71)
    sensors = ['temperature', 'current', 'currentTXPower', 'currentRXPower',
               *[f'currentMultiRXPower{i}' for i in range(1,5)],
               *[f'currentMultiTXPower{i}' for i in range(1,5)]]
    rows = []
    for i in range(30):
        faulty = i % 2
        values = rng.normal(size=(48,12))*.02 + 1 + faulty*.02
        values[:,0] += 35
        frame = pd.DataFrame(values, columns=sensors)
        frame.insert(0, 'timestamp', 1700000000 + np.arange(48)*300 + i*86400)
        frame['anomaly'] = 0
        if faulty:
            frame.loc[47, 'anomaly'] = 1
        name = f'synthetic_{i:03d}.csv'
        frame.to_csv(data_dir/name, index=False)
        rows.append(dict(file_name=name, Label=faulty))
    index_path = root/'index.csv'
    pd.DataFrame(rows).to_csv(index_path, index=False)
    return data_dir, index_path


def run_smoke(output, device='cpu'):
    from .experiment import run_experiment
    from .model import VARIANTS
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    data_dir, index_path = generate_data(output)
    run_experiment(data_dir,index_path,output/'experiments',VARIANTS,device=device,smoke=True)
    print('PASS: synthetic training, five component variants, validation selection, test inference and seven metrics.')
