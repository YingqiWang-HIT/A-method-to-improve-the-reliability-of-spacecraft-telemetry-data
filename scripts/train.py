from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from hstgat_cd.utils.io import load_yaml, load_json, save_json
from hstgat_cd.utils.seed import set_seed
from hstgat_cd.data.preprocessing import load_csv_timeseries, ZScoreNormalizer
from hstgat_cd.data.dataset import create_irregular_windows, split_bundle, TelemetryWindowDataset
from hstgat_cd.utils.graph_builder import build_hierarchical_graph
from hstgat_cd.models.hstgat_cd import HSTGATCD
from hstgat_cd.utils.losses import masked_mse_loss, subsystem_soft_sharing_loss
from hstgat_cd.utils.metrics import masked_mae, masked_rmse, masked_mape


def choose_device(name: str):
    if name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def evaluate_model(model, loader, device):
    model.eval()
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            obs_mask = batch['obs_mask'].to(device)
            target_mask = batch['target_mask'].to(device)
            dt = batch['dt'].to(device)
            pred = model(x, dt, obs_mask)
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
            masks.append(target_mask.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target = np.concatenate(targets, axis=0)
    mask = np.concatenate(masks, axis=0)
    return {
        'mae': float(masked_mae(pred, target, mask)),
        'rmse': float(masked_rmse(pred, target, mask)),
        'mape': float(masked_mape(pred, target, mask)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--layout_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='run_output')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg['seed'])
    device = choose_device(cfg['device'])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'plots').mkdir(exist_ok=True)

    timestamps, values, feature_names = load_csv_timeseries(args.data_path, cfg['data']['timestamp_col'])
    layout = load_json(args.layout_path)

    norm = ZScoreNormalizer().fit(values)
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
    train_bundle, val_bundle, test_bundle = split_bundle(bundle, cfg['data']['val_ratio'], cfg['data']['test_ratio'])

    graph_dict = build_hierarchical_graph(values_norm, feature_names, layout, cfg)
    model = HSTGATCD(num_nodes=len(feature_names), graph_dict=graph_dict, config=cfg).to(device)

    train_loader = DataLoader(TelemetryWindowDataset(train_bundle), batch_size=cfg['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(TelemetryWindowDataset(val_bundle), batch_size=cfg['train']['batch_size'], shuffle=False)
    test_loader = DataLoader(TelemetryWindowDataset(test_bundle), batch_size=cfg['train']['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    best_val = float('inf')
    patience_left = cfg['train']['patience']
    history = []

    for epoch in range(1, cfg['train']['epochs'] + 1):
        model.train()
        batch_losses = []
        for batch in train_loader:
            x = batch['x'].to(device)
            y = batch['y'].to(device)
            obs_mask = batch['obs_mask'].to(device)
            target_mask = batch['target_mask'].to(device)
            dt = batch['dt'].to(device)

            pred = model(x, dt, obs_mask)
            recon_loss = masked_mse_loss(pred, y, target_mask)
            soft_loss = subsystem_soft_sharing_loss(model, graph_dict['subsystem_indices'])
            loss = cfg['loss']['recon_weight'] * recon_loss + cfg['loss']['soft_sharing_weight'] * soft_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['train']['grad_clip'])
            optimizer.step()
            batch_losses.append(loss.item())

        val_metrics = evaluate_model(model, val_loader, device)
        train_loss = float(np.mean(batch_losses)) if batch_losses else float('nan')
        history.append({'epoch': epoch, 'train_loss': train_loss, **{f'val_{k}': v for k, v in val_metrics.items()}})
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_rmse={val_metrics['rmse']:.6f} | val_mae={val_metrics['mae']:.6f}")

        if val_metrics['rmse'] < best_val:
            best_val = val_metrics['rmse']
            patience_left = cfg['train']['patience']
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'feature_names': feature_names,
                'graph_dict': graph_dict,
                'normalizer_mean': norm.mean_,
                'normalizer_std': norm.std_,
            }, out_dir / 'best_model.pt')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print('Early stopping triggered.')
                break

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': cfg,
        'feature_names': feature_names,
        'graph_dict': graph_dict,
        'normalizer_mean': norm.mean_,
        'normalizer_std': norm.std_,
    }, out_dir / 'last_model.pt')

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / 'train_history.csv', index=False)

    ckpt = torch.load(out_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate_model(model, test_loader, device)
    save_json(test_metrics, out_dir / 'metrics.json')

    plt.figure(figsize=(6, 4))
    plt.plot(hist_df['epoch'], hist_df['train_loss'], label='Train loss')
    plt.plot(hist_df['epoch'], hist_df['val_rmse'], label='Val RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / RMSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'loss_curve.png', dpi=200)
    plt.close()

    with open(out_dir / 'run_summary.txt', 'w', encoding='utf-8') as f:
        f.write(json.dumps(test_metrics, indent=2))
    print('Training complete.')
    print('Test metrics:', test_metrics)


if __name__ == '__main__':
    main()
