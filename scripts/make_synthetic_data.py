from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='demo_data')
    parser.add_argument('--length', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    T = args.length
    t = np.arange(T, dtype=float)

    data = {
        'timestamp': t,
        'att_1': np.sin(0.03 * t) + 0.1 * rng.normal(size=T),
        'att_2': np.sin(0.03 * t + 0.5) + 0.08 * rng.normal(size=T),
        'att_3': np.cos(0.02 * t) * (1 + 0.001 * t) + 0.08 * rng.normal(size=T),
        'thermal_1': 0.002 * t + 0.2 * np.sin(0.01 * t) + 0.05 * rng.normal(size=T),
        'thermal_2': 0.003 * t + 0.1 * np.cos(0.012 * t) + 0.05 * rng.normal(size=T),
        'power_1': (np.sin(0.05 * t) > 0).astype(float) + 0.05 * rng.normal(size=T),
        'power_2': np.sign(np.sin(0.04 * t)) + 0.05 * rng.normal(size=T),
        'orbit_1': np.sin(0.008 * t) + 0.5 * np.sin(0.05 * t) + 0.03 * rng.normal(size=T),
        'orbit_2': np.cos(0.008 * t) + 0.03 * rng.normal(size=T),
    }

    # introduce small drift and abrupt disturbances
    for k in ['att_1', 'thermal_1', 'power_1', 'orbit_1']:
        idx = slice(T // 3, T // 3 + 120)
        data[k][idx] += np.linspace(0, 0.8, 120)
    for k in ['att_2', 'power_2']:
        idx = slice(2 * T // 3, 2 * T // 3 + 80)
        data[k][idx] -= 0.7

    # true missing values in raw data
    for key in list(data.keys())[1:]:
        miss = rng.choice(T, size=int(0.02 * T), replace=False)
        arr = data[key].copy()
        arr[miss] = np.nan
        data[key] = arr

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(out_dir / 'telemetry.csv', index=False)

    layout = {
        'subsystems': {
            'attitude': ['att_1', 'att_2', 'att_3'],
            'thermal': ['thermal_1', 'thermal_2'],
            'power': ['power_1', 'power_2'],
            'orbit': ['orbit_1', 'orbit_2'],
        }
    }
    with open(out_dir / 'subsystem_layout.json', 'w', encoding='utf-8') as f:
        json.dump(layout, f, indent=2)

    print(f'Saved synthetic dataset to {out_dir}')


if __name__ == '__main__':
    main()
