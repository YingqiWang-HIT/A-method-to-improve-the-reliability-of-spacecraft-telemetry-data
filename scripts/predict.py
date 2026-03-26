from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from hstgat_cd.utils.io import load_yaml
from hstgat_cd.data.preprocessing import load_csv_timeseries, ZScoreNormalizer
from hstgat_cd.data.dataset import create_irregular_windows, TelemetryWindowDataset
from hstgat_cd.models.hstgat_cd import HSTGATCD


def choose_device(name: str):
    if name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--layout_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = choose_device(cfg['device'])
    ckpt = torch.load(args.checkpoint, map_location=device)

    timestamps, values, feature_names = load_csv_timeseries(args.data_path, cfg['data']['timestamp_col'])
    norm = ZScoreNormalizer()
    norm.mean_ = ckpt['normalizer_mean']
    norm.std_ = ckpt['normalizer_std']
    values_norm = norm.transform(values)

    bundle = create_irregular_windows(
        timestamps=timestamps,
        values=values_norm,
        window_size=cfg['data']['window_size'],
        stride=cfg['data']['stride'],
        mask_rate=cfg['data']['mask_rate'],
        min_observed_ratio=cfg['data']['min_observed_ratio'],
        seed=cfg['seed'],
    )
    loader = DataLoader(TelemetryWindowDataset(bundle), batch_size=cfg['train']['batch_size'], shuffle=False)

    model = HSTGATCD(num_nodes=len(feature_names), graph_dict=ckpt['graph_dict'], config=cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch['x'].to(device), batch['dt'].to(device), batch['obs_mask'].to(device))
            preds.append(pred.cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    pred_inv = norm.inverse_transform(pred.reshape(-1, pred.shape[-1])).reshape(pred.shape)

    rows = []
    for w in range(pred_inv.shape[0]):
        for t in range(pred_inv.shape[1]):
            row = {'window_id': w, 'step_id': t}
            for i, name in enumerate(feature_names):
                row[name] = pred_inv[w, t, i]
            rows.append(row)
    pd.DataFrame(rows).to_csv(args.output_file, index=False)
    print(f'Saved predictions to {args.output_file}')


if __name__ == '__main__':
    main()
