"""Runner-permutation-equivariant actor-critic with a factorised action head.

Each runner is a token (its own features + the global market features). A small
transformer lets runners attend to each other (money moving off one horse onto
another is the main pre-race signal), then shared heads emit that runner's
action. A learned market token carries the value estimate. Field size can be
anything up to R_MAX; padded runners are masked out everywhere.

The per-runner action is factorised as
    type  in {NOOP, CLOSE, CANCEL_ENTRY, OPEN}
    side, entry mode, stake, take-profit   (only used when type == OPEN)
which is far easier to learn than one flat softmax over every combination:
"back this horse" is a single decision regardless of the stake/exit chosen.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import ENTRY_MODES, N_FIXED, EnvConfig
from .features import N_GLOBAL_FEATURES, N_RUNNER_FEATURES

TYPE_OPEN = 3


class ActionCodec:
    """Converts factorised components [..., 5] <-> flat env action ids."""

    def __init__(self, cfg: EnvConfig):
        self.n_st, self.n_tp = len(cfg.stakes), len(cfg.tp_ticks)
        self.sizes = (N_FIXED + 1, 2, len(ENTRY_MODES), self.n_st, self.n_tp)

    def to_flat(self, comp: np.ndarray) -> np.ndarray:
        typ, side, mode, st, tp = (comp[..., i] for i in range(5))
        flat = N_FIXED + ((side * len(ENTRY_MODES) + mode) * self.n_st + st) * self.n_tp + tp
        return np.where(typ == TYPE_OPEN, flat, typ).astype(np.int64)


class RunnerTransformerPolicy(nn.Module):
    def __init__(self, cfg: EnvConfig, d_model: int = 96, n_layers: int = 2, n_heads: int = 4,
                 noop_bias: float = 3.0, close_bias: float = -3.0):
        super().__init__()
        self.codec = ActionCodec(cfg)
        self.runner_in = nn.Sequential(nn.Linear(N_RUNNER_FEATURES + N_GLOBAL_FEATURES, d_model), nn.GELU(),
                                       nn.Linear(d_model, d_model))
        self.market_in = nn.Sequential(nn.Linear(N_GLOBAL_FEATURES, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 2 * d_model, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.trunk = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(d_model, n) for n in self.codec.sizes])
        # per-runner critic: each runner is credited with its own P&L
        self.v = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        with torch.no_grad():
            for h in self.heads:
                h.weight.mul_(0.01)
                h.bias.zero_()
            # start mostly idle; exploration is sparse rather than a firehose of orders.
            # CLOSE / CANCEL start rare: brackets already manage exits, and random
            # closes would destroy exactly the trades exploration needs to evaluate.
            self.heads[0].bias[0] = noop_bias
            self.heads[0].bias[1] = close_bias
            self.heads[0].bias[2] = close_bias
            self.v[-1].weight.mul_(0.01)
            self.v[-1].bias.zero_()

    def forward(self, runners, glob, mask):
        B, R, _ = runners.shape
        mask = mask.bool()
        x = self.runner_in(torch.cat([runners, glob[:, None, :].expand(B, R, -1)], -1))
        m = self.market_in(glob)[:, None, :]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask], 1)
        h = self.norm(self.encoder(torch.cat([m, x], 1), src_key_padding_mask=pad))
        hm, hr = h[:, 0], h[:, 1:]
        z = self.trunk(hr)
        logits = [head(z) for head in self.heads]
        value = self.v(torch.cat([hr, hm[:, None, :].expand_as(hr)], -1)).squeeze(-1)  # [B,R]
        return logits, value

    def dist(self, runners, glob, mask):
        logits, value = self.forward(runners, glob, mask)
        return [Categorical(logits=l) for l in logits], value

    @staticmethod
    def sample(dists, deterministic=False):
        return torch.stack([d.probs.argmax(-1) if deterministic else d.sample() for d in dists], -1)

    @staticmethod
    def runner_logprob(dists, comp):
        """comp [B,R,5] -> [B,R] log-prob of each runner's action."""
        is_open = (comp[..., 0] == TYPE_OPEN).float()
        lp = dists[0].log_prob(comp[..., 0])
        for i in range(1, 5):
            lp = lp + is_open * dists[i].log_prob(comp[..., i])
        return lp

    @staticmethod
    def runner_entropy(dists):
        p_open = dists[0].probs[..., TYPE_OPEN]
        return dists[0].entropy() + p_open * sum(d.entropy() for d in dists[1:])

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False, device="cpu"):
        r = torch.as_tensor(obs["runners"], dtype=torch.float32, device=device)
        g = torch.as_tensor(obs["global"], dtype=torch.float32, device=device)
        m = torch.as_tensor(obs["mask"], dtype=torch.float32, device=device)
        squeeze = r.dim() == 2
        if squeeze:
            r, g, m = r[None], g[None], m[None]
        dists, value = self.dist(r, g, m)
        comp = self.sample(dists, deterministic) * m.long()[..., None]
        flat = self.codec.to_flat(comp.cpu().numpy())
        if squeeze:
            return flat[0], comp[0].cpu().numpy(), value[0].cpu().numpy()
        return flat, comp.cpu().numpy(), value.cpu().numpy()
