"""
ppo_agent.py
CA-MORL: Proximal Policy Optimization Agent
============================================
Implements PPO with clipped surrogate objective and GAE advantage estimation.
Compatible with the CA-MORL environment wrapper.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


# ─────────────────────────────────────────────────────────────────────────────
# Actor-Critic Network
# ─────────────────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared-trunk MLP with separate actor and critic heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor_mean  = nn.Linear(hidden, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, obs):
        h = self.trunk(obs)
        mean   = self.actor_mean(h)
        std    = self.actor_logstd.exp().expand_as(mean)
        dist   = Normal(mean, std)
        value  = self.critic(h).squeeze(-1)
        return dist, value

    def get_action(self, obs):
        dist, value = self(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(self, obs, action):
        dist, value = self(obs)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, value, entropy


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Fixed-size rollout buffer for on-policy PPO."""

    def __init__(self, size: int, obs_dim: int, act_dim: int, device: str = 'cpu'):
        self.size = size
        self.device = device
        self.obs      = torch.zeros(size, obs_dim)
        self.actions  = torch.zeros(size, act_dim)
        self.rewards  = torch.zeros(size)
        self.dones    = torch.zeros(size)
        self.log_probs = torch.zeros(size)
        self.values   = torch.zeros(size)
        self.ptr = 0

    def store(self, obs, action, reward, done, log_prob, value):
        i = self.ptr % self.size
        self.obs[i]      = torch.as_tensor(obs,      dtype=torch.float32)
        self.actions[i]  = torch.as_tensor(action,   dtype=torch.float32)
        self.rewards[i]  = torch.as_tensor(reward,   dtype=torch.float32)
        self.dones[i]    = torch.as_tensor(done,     dtype=torch.float32)
        self.log_probs[i] = torch.as_tensor(log_prob, dtype=torch.float32)
        self.values[i]   = torch.as_tensor(value,    dtype=torch.float32)
        self.ptr += 1

    def full(self):
        return self.ptr >= self.size

    def compute_gae(self, last_value: float, gamma: float = 0.99,
                    gae_lambda: float = 0.95):
        """Compute Generalised Advantage Estimates and returns."""
        advantages = torch.zeros(self.size)
        last_adv   = 0.0
        last_val   = last_value
        for t in reversed(range(self.size)):
            mask   = 1.0 - self.dones[t].item()
            delta  = (self.rewards[t].item()
                      + gamma * last_val * mask
                      - self.values[t].item())
            last_adv = delta + gamma * gae_lambda * mask * last_adv
            advantages[t] = last_adv
            last_val = self.values[t].item()
        returns = advantages + self.values
        return advantages, returns

    def get_batches(self, advantages, returns, batch_size: int = 64):
        """Yield mini-batches for PPO update."""
        idx = torch.randperm(self.size)
        for start in range(0, self.size, batch_size):
            b = idx[start:start + batch_size]
            yield (self.obs[b], self.actions[b], self.log_probs[b],
                   advantages[b], returns[b])

    def reset(self):
        self.ptr = 0


# ─────────────────────────────────────────────────────────────────────────────
# PPO Agent
# ─────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    PPO agent for CA-MORL.

    Parameters
    ----------
    obs_dim      : observation dimensionality (11 for CA-MORL)
    act_dim      : action dimensionality (2 for CA-MORL)
    lr           : Adam learning rate (default 3e-4)
    clip_ratio   : PPO clipping epsilon (default 0.2)
    gamma        : discount factor (default 0.99)
    gae_lambda   : GAE lambda (default 0.95)
    update_epochs: PPO update epochs per rollout (default 10)
    batch_size   : mini-batch size (default 64)
    buffer_size  : rollout buffer size (default 2048)
    vf_coef      : value function loss coefficient (default 0.5)
    ent_coef     : entropy bonus coefficient (default 0.01)
    max_grad_norm: gradient clipping (default 0.5)
    hidden       : hidden layer width (default 256)
    device       : torch device
    """

    def __init__(self, obs_dim, act_dim,
                 lr=3e-4, clip_ratio=0.2, gamma=0.99, gae_lambda=0.95,
                 update_epochs=10, batch_size=64, buffer_size=2048,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
                 hidden=256, device='cpu'):

        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_ratio   = clip_ratio
        self.update_epochs = update_epochs
        self.batch_size   = batch_size
        self.vf_coef      = vf_coef
        self.ent_coef     = ent_coef
        self.max_grad_norm = max_grad_norm
        self.device       = device

        self.net = ActorCritic(obs_dim, act_dim, hidden).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer(buffer_size, obs_dim, act_dim, device)

        self._total_steps = 0
        self.loss_history = []

    # ── Interaction ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, obs: np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        action, log_prob, value = self.net.get_action(obs_t)
        return (action.cpu().numpy(),
                log_prob.cpu().item(),
                value.cpu().item())

    def store_transition(self, obs, action, reward, done, log_prob, value):
        self.buffer.store(obs, action, reward, done, log_prob, value)
        self._total_steps += 1

    # ── Update ───────────────────────────────────────────────────────────────

    def update(self, last_obs: np.ndarray):
        """Run PPO update over the current rollout buffer."""
        with torch.no_grad():
            obs_t = torch.as_tensor(last_obs, dtype=torch.float32).to(self.device)
            _, last_value = self.net(obs_t)
            last_value = last_value.cpu().item()

        advantages, returns = self.buffer.compute_gae(last_value,
                                                       self.gamma,
                                                       self.gae_lambda)
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        epoch_losses = []
        for _ in range(self.update_epochs):
            for obs_b, act_b, old_lp_b, adv_b, ret_b in \
                    self.buffer.get_batches(advantages, returns, self.batch_size):
                obs_b  = obs_b.to(self.device)
                act_b  = act_b.to(self.device)
                old_lp_b = old_lp_b.to(self.device)
                adv_b  = adv_b.to(self.device)
                ret_b  = ret_b.to(self.device)

                log_prob, value, entropy = self.net.evaluate(obs_b, act_b)
                ratio = (log_prob - old_lp_b).exp()

                # Clipped surrogate objective
                surr1 = ratio * adv_b
                surr2 = ratio.clamp(1 - self.clip_ratio,
                                    1 + self.clip_ratio) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = (ret_b - value).pow(2).mean()

                # Total loss
                loss = (policy_loss
                        + self.vf_coef * value_loss
                        - self.ent_coef * entropy.mean())

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                epoch_losses.append(loss.item())

        self.buffer.reset()
        mean_loss = np.mean(epoch_losses)
        self.loss_history.append(mean_loss)
        return mean_loss

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            'net_state': self.net.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'total_steps': self._total_steps,
        }, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck['net_state'])
        self.optimizer.load_state_dict(ck['opt_state'])
        self._total_steps = ck.get('total_steps', 0)
