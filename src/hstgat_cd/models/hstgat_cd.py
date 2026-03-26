from __future__ import annotations
import torch
import torch.nn as nn
from .vlstm_sde import VLSTMSDEEncoder
from .graph_attention import MultiHeadGraphAttention


class HSTGATCD(nn.Module):
    def __init__(self, num_nodes: int, graph_dict: dict, config: dict):
        super().__init__()
        mcfg = config['model']
        self.num_nodes = num_nodes
        self.graph_dict = graph_dict
        hidden_dim = mcfg['hidden_dim']
        gat_hidden_dim = mcfg['gat_hidden_dim']

        self.temporal_encoder = VLSTMSDEEncoder(
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            temporal_layers=mcfg['temporal_layers'],
            brownian_std=mcfg['brownian_std'],
            hidden_decay=mcfg['hidden_decay'],
        )
        self.local_gat = MultiHeadGraphAttention(hidden_dim, gat_hidden_dim, num_heads=mcfg['num_heads'], dropout=mcfg['dropout'])
        self.cross_gat = MultiHeadGraphAttention(gat_hidden_dim, gat_hidden_dim, num_heads=mcfg['num_heads'], dropout=mcfg['dropout'])
        self.global_gat = MultiHeadGraphAttention(gat_hidden_dim, gat_hidden_dim, num_heads=mcfg['num_heads'], dropout=mcfg['dropout'])
        self.norm = nn.LayerNorm(gat_hidden_dim) if mcfg.get('use_layernorm', True) else nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(gat_hidden_dim, gat_hidden_dim),
            nn.ReLU(),
            nn.Linear(gat_hidden_dim, 1),
        )

        self.register_buffer('A_local', torch.tensor(graph_dict['A_local'], dtype=torch.float32))
        self.register_buffer('A_cross', torch.tensor(graph_dict['A_cross'], dtype=torch.float32))
        self.register_buffer('A_global_nodes', torch.tensor(graph_dict['A_global_nodes'], dtype=torch.float32))

    def forward(self, x, dt, obs_mask):
        z = self.temporal_encoder(x, dt, obs_mask)
        z = self.local_gat(z, self.A_local)
        z = self.cross_gat(z, self.A_cross) + z
        z = self.global_gat(z, self.A_global_nodes) + z
        z = self.norm(z)
        y = self.head(z).squeeze(-1)
        return y
