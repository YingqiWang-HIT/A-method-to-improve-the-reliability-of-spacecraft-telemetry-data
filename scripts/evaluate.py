from __future__ import annotations
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from hstgat_cd.utils.io import load_yaml, load_json
from hstgat_cd.data.preprocessing import load_csv_timeseries, ZScoreNormalizer
from hstgat_cd.data.dataset import create_irregular_windows, split_bundle, TelemetryWindowDataset
from hstgat_cd.models.hstgat_cd import HSTGATCD
from hstgat_cd.utils.metrics import masked_mae, masked_rmse, masked_mape


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
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = choose_device(cfg['device'])
    ckpt = torch.load(args.checkpoint, map_location=device)

    timestamps, values, feature_names = load_csv_timeseries(args.data_path, cfg['data']['timestamp_col'])
    _ = load_json(args.layout_path)
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
    _, _, test_bundle = split_bundle(bundle, cfg['data']['val_ratio'], cfg['data']['test_ratio'])
    loader = DataLoader(TelemetryWindowDataset(test_bundle), batch_size=cfg['train']['batch_size'], shuffle=False)

    model = HSTGATCD(num_nodes=len(feature_names), graph_dict=ckpt['graph_dict'], config=cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    preds, targets, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred = model(batch['x'].to(device), batch['dt'].to(device), batch['obs_mask'].to(device))
            preds.append(pred.cpu().numpy())
            targets.append(batch['y'].numpy())
            masks.append(batch['target_mask'].numpy())

    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    mask = np.concatenate(masks, axis=0)

    metrics = {
        'MAE': float(masked_mae(pred, target, mask)),
        'RMSE': float(masked_rmse(pred, target, mask)),
        'MAPE': float(masked_mape(pred, target, mask)),
    }
    print(metrics)


if __name__ == '__main__':
    main()
