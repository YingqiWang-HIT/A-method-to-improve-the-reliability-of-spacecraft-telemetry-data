from __future__ import annotations
import torch
import torch.nn as nn


class NodewiseLinear(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_nodes, hidden_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_nodes, hidden_dim))

    def forward(self, x):
        # x: [B, N]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class VLSTMSDECell(nn.Module):
    def __init__(self, hidden_dim: int, brownian_std: float = 0.05, hidden_decay: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.brownian_std = brownian_std
        self.hidden_decay = hidden_decay

        self.f_gate = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.i_gate = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.o_gate = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.c_gate = nn.Linear(hidden_dim * 2 + 1, hidden_dim)

        self.mod_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev, c_prev, dt):
        # x_t/h_prev/c_prev: [B, N, H], dt: [B, N, 1]
        gate_in = torch.cat([x_t, h_prev, dt], dim=-1)
        f = torch.sigmoid(self.f_gate(gate_in))
        i = torch.sigmoid(self.i_gate(gate_in))
        o = torch.sigmoid(self.o_gate(gate_in))
        c_tilde = torch.tanh(self.c_gate(gate_in))

        mod = self.mod_mlp(gate_in)
        A, omega, phi = torch.chunk(mod, 3, dim=-1)
        z_t = A + A * torch.sin(omega * dt + phi)

        decay = -(1.0 - f) / dt.clamp_min(1e-4)
        inject = i / dt.clamp_min(1e-4)
        brownian = torch.randn_like(c_prev) * self.brownian_std * torch.sqrt(dt.clamp_min(1e-4))

        c = c_prev + (decay * c_prev + inject * c_tilde + z_t) * dt + brownian
        h = h_prev + (o * torch.tanh(c) - self.hidden_decay * h_prev) * dt
        y = self.out_proj(h)
        return y, h, c


class VLSTMSDEEncoder(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int, temporal_layers: int = 2, brownian_std: float = 0.05, hidden_decay: float = 0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.temporal_layers = temporal_layers
        self.node_linears = NodewiseLinear(num_nodes, hidden_dim)
        self.layers = nn.ModuleList([
            VLSTMSDECell(hidden_dim, brownian_std=brownian_std, hidden_decay=hidden_decay)
            for _ in range(temporal_layers)
        ])

    def forward(self, x, dt, obs_mask):
        # x, dt, obs_mask: [B, T, N]
        B, T, N = x.shape
        device = x.device
        h_states = [torch.zeros(B, N, self.hidden_dim, device=device) for _ in range(self.temporal_layers)]
        c_states = [torch.zeros(B, N, self.hidden_dim, device=device) for _ in range(self.temporal_layers)]
        outputs = []

        for t in range(T):
            x_t = self.node_linears(x[:, t, :])
            # mark missing observations but keep zero-filled input
            x_t = x_t * obs_mask[:, t, :, None]
            dt_t = dt[:, t, :, None]
            layer_in = x_t
            for l, layer in enumerate(self.layers):
                y, h, c = layer(layer_in, h_states[l], c_states[l], dt_t)
                h_states[l], c_states[l] = h, c
                layer_in = y
            outputs.append(layer_in)

        return torch.stack(outputs, dim=1)  # [B, T, N, H]
