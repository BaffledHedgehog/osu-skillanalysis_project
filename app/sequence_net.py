"""Small temporal CNN, masked geometry pretraining and map-level residual heads."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .sequence import FEATURES


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation)
        self.norm = nn.GroupNorm(4, channels)
        self.drop = nn.Dropout(.1)

    def forward(self, x, mask):
        return (x + self.drop(F.gelu(self.norm(self.conv(x))))) * mask[:, None]


class PatternEncoder(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.input = nn.Conv1d(len(FEATURES), channels, 1)
        self.blocks = nn.ModuleList([TemporalBlock(channels, d) for d in (1, 2, 4)])
        self.reconstruction = nn.Conv1d(channels, len(FEATURES), 1)

    def forward(self, notes, mask):
        x = F.gelu(self.input(notes)) * mask[:, None]
        for block in self.blocks:
            x = block(x, mask)
        return x

    def reconstruct(self, notes, mask):
        return self.reconstruction(self(notes, mask))


class PatternModel(nn.Module):
    def __init__(self, context_size, channels=32, sequence=True):
        super().__init__()
        self.encoder = PatternEncoder(channels)
        self.sequence = sequence
        self.local = nn.Linear(channels, 1)
        self.attention = nn.Linear(channels * 2, 1)
        self.body = nn.Sequential(nn.Linear(context_size + channels * 6, 64), nn.GELU(), nn.Dropout(.15),
                                  nn.Linear(64, 32), nn.GELU())
        self.pass_delta = nn.Linear(32, 1)
        self.accuracy_delta = nn.Linear(32, 1)
        # The initial prediction is exactly the training-fold baseline. Learning
        # is allowed to improve it, never disguised as an already trained model.
        nn.init.zeros_(self.pass_delta.weight); nn.init.zeros_(self.pass_delta.bias)
        nn.init.zeros_(self.accuracy_delta.weight); nn.init.zeros_(self.accuracy_delta.bias)

    def forward(self, notes, mask, bag_ids, context, base_pass_logits, base_accuracy):
        if self.sequence:
            chunks, local_scores = self.encode_chunks(notes, mask)
        else:
            chunks = torch.zeros((len(notes), self.encoder.input.out_channels * 2), device=context.device)
            local_scores = torch.zeros(len(notes), device=context.device)
        return (*self.from_chunks(chunks, bag_ids, context, base_pass_logits, base_accuracy), local_scores)

    def encode_chunks(self, notes, mask):
        x = self.encoder(notes, mask)
        avg = (x * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1)
        maximum = x.masked_fill(~mask[:, None], -1e4).amax(-1)
        return torch.cat([avg, maximum], dim=1), self.local(avg).squeeze(-1)

    def from_chunks(self, chunks, bag_ids, context, base_pass_logits, base_accuracy):
        batch_size = len(context)
        if self.sequence:
            pooled = []
            for i in range(batch_size):
                rows = chunks[bag_ids == i]
                attention = torch.softmax(self.attention(rows).squeeze(-1), 0)
                pooled.append(torch.cat([rows.mean(0), rows.amax(0), (rows * attention[:, None]).sum(0)]))
            embedding = torch.stack(pooled)
        else:
            embedding = torch.zeros((batch_size, self.encoder.input.out_channels * 6), device=context.device)
        hidden = self.body(torch.cat([context, embedding], dim=1))
        pass_logits = base_pass_logits + self.pass_delta(hidden).squeeze(-1)
        accuracy = (base_accuracy + .1 * self.accuracy_delta(hidden).squeeze(-1)).clamp(0, 1)
        return pass_logits, accuracy


class LocalPatternModel(nn.Module):
    """Replay-supervised note risks, separate from map pass/accuracy heads."""
    def __init__(self, context_size, channels=32, targets=2):
        super().__init__()
        self.encoder = PatternEncoder(channels)
        self.context = nn.Linear(context_size, 16)
        self.head = nn.Sequential(nn.Conv1d(channels + 16, 32, 1), nn.GELU(),
                                  nn.Dropout(.1), nn.Conv1d(32, targets, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, notes, mask, context, baseline_logits):
        encoded = self.encoder(notes, mask)
        player = F.gelu(self.context(context))[:, :, None].expand(-1, -1, notes.shape[-1])
        return baseline_logits + self.head(torch.cat([encoded, player], dim=1)) * mask[:, None]


def save_weights(model, path):
    """Plain numeric NPZ, no pickle or executable objects in model artifacts."""
    arrays = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    np.savez_compressed(path, **arrays)


def load_weights(model, path):
    with np.load(path, allow_pickle=False) as values:
        expected = model.state_dict()
        if set(values.files) != set(expected):
            raise ValueError('Invalid CNN weight names.')
        state = {}
        for name, tensor in expected.items():
            value = values[name]
            if tuple(value.shape) != tuple(tensor.shape) or not np.isfinite(value).all():
                raise ValueError('Invalid CNN weight shape or values.')
            state[name] = torch.from_numpy(value.copy())
    model.load_state_dict(state)
    return model
