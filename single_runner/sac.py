"""Self-contained black-box Soft Actor-Critic (PyTorch).

The agent knows nothing about betting: it sees a flat observation vector and
emits a 6-dim action in [-1, 1]. Components:
  * tanh-squashed Gaussian actor, twin Q critics + Polyak target nets
  * automatic entropy tuning (target entropy = -action_dim)
  * running observation normalisation (frozen at evaluation)
  * float16 replay storage (obs are ~1.2k dims) to keep memory sane
"""
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


@dataclass
class SACConfig:
    hidden: tuple = (512, 512, 256)
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 300_000
    learning_starts: int = 5_000
    updates_per_step: int = 1
    init_alpha: float = 0.1
    target_entropy: float = None      # default -action_dim
    grad_clip: float = 10.0
    obs_clip: float = 10.0
    device: str = "auto"


class RunningNorm:
    def __init__(self, dim, clip=10.0):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, x):
        x = np.atleast_2d(x)
        bm, bv, bn = x.mean(0), x.var(0), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bn
        self.mean = self.mean + delta * bn / tot
        self.var = (self.var * self.count + bv * bn + delta ** 2 * self.count * bn / tot) / tot
        self.count = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count, "clip": self.clip}

    def load_state_dict(self, d):
        self.mean, self.var, self.count, self.clip = d["mean"], d["var"], d["count"], d["clip"]


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), np.float16)
        self.next_obs = np.zeros((size, obs_dim), np.float16)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.size, self.ptr, self.full = size, 0, False

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n):
        idx = np.random.randint(0, len(self), n)
        return (self.obs[idx].astype(np.float32), self.act[idx], self.rew[idx],
                self.next_obs[idx].astype(np.float32), self.done[idx])


def mlp(inp, out, hidden):
    layers, d = [], inp
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)

    def forward(self, obs, deterministic=False):
        mu, log_std = self.net(obs).chunk(2, -1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        if deterministic:
            return torch.tanh(mu), None
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SACAgent:
    def __init__(self, obs_dim, act_dim, config=None):
        self.cfg = c = config or SACConfig()
        if c.device == "auto":
            c.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(c.device)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.actor = Actor(obs_dim, act_dim, c.hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, c.hidden).to(self.device)
        self.critic_targ = Critic(obs_dim, act_dim, c.hidden).to(self.device)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)
        self.log_alpha = torch.tensor(np.log(c.init_alpha), device=self.device, requires_grad=True)
        self.target_entropy = c.target_entropy if c.target_entropy is not None else -float(act_dim)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=c.lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=c.lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=c.lr)
        self.norm = RunningNorm(obs_dim, c.obs_clip)
        self.n_updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """obs: raw (un-normalised) observation(s)."""
        o = torch.as_tensor(self.norm(obs), device=self.device).float()
        single = o.dim() == 1
        a, _ = self.actor(o.unsqueeze(0) if single else o, deterministic)
        a = a.cpu().numpy()
        return a[0] if single else a

    def update(self, buf):
        c = self.cfg
        o, a, r, o2, d = buf.sample(c.batch_size)
        o = torch.as_tensor(self.norm(o), device=self.device)
        o2 = torch.as_tensor(self.norm(o2), device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        d = torch.as_tensor(d, device=self.device)
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q1t, q2t = self.critic_targ(o2, a2)
            target = r + c.gamma * (1 - d) * (torch.min(q1t, q2t) - alpha * logp2)
        q1, q2 = self.critic(o, a)
        loss_q = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_critic.zero_grad()
        loss_q.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), c.grad_clip)
        self.opt_critic.step()

        for p in self.critic.parameters():
            p.requires_grad_(False)
        pi, logp = self.actor(o)
        q1p, q2p = self.critic(o, pi)
        loss_pi = (alpha * logp - torch.min(q1p, q2p)).mean()
        self.opt_actor.zero_grad()
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_clip)
        self.opt_actor.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_targ.parameters()):
                pt.mul_(1 - c.tau).add_(c.tau * p)
        self.n_updates += 1
        return {"loss_q": loss_q.item(), "loss_pi": loss_pi.item(), "alpha": alpha.item(),
                "entropy": -logp.mean().item(), "q": q1.mean().item()}

    def save(self, path, extra=None):
        cfg = asdict(self.cfg)
        torch.save({
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "critic_targ": self.critic_targ.state_dict(), "log_alpha": self.log_alpha.detach().cpu(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "opt_alpha": self.opt_alpha.state_dict(), "norm": self.norm.state_dict(),
            "obs_dim": self.obs_dim, "act_dim": self.act_dim, "sac_config": cfg,
            "n_updates": self.n_updates, "extra": extra or {},
        }, path)

    @classmethod
    def load(cls, path, device="auto"):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = SACConfig(**{**ck["sac_config"], "device": device})
        cfg.hidden = tuple(cfg.hidden)
        agent = cls(ck["obs_dim"], ck["act_dim"], cfg)
        agent.actor.load_state_dict(ck["actor"])
        agent.critic.load_state_dict(ck["critic"])
        agent.critic_targ.load_state_dict(ck["critic_targ"])
        with torch.no_grad():
            agent.log_alpha.copy_(ck["log_alpha"].to(agent.device))
        agent.opt_actor.load_state_dict(ck["opt_actor"])
        agent.opt_critic.load_state_dict(ck["opt_critic"])
        agent.opt_alpha.load_state_dict(ck["opt_alpha"])
        agent.norm.load_state_dict(ck["norm"])
        agent.n_updates = ck["n_updates"]
        return agent, ck.get("extra", {})
