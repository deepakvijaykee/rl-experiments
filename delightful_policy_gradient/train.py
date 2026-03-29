"""Training loop, experience queue, gradient diagnostics, and CLI entry point."""

import argparse
import dataclasses
import time
from collections import deque
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F

from . import losses as L
from .tasks import Batch, MNISTBandit, TokenReversal, LMBandit


# ── Experience Queue ─────────────────────────────────────────────────────────


class ExperienceQueue:
    """Ring buffer of pre-sampled Batches for delayed training.

    Replaces StalenessBuffer (which stored model state_dicts). Scales with
    batch size, not model size — O(D × B) vs O(D × |θ|).

    Staleness semantics: at step t, the training batch has actor_log_probs
    from D steps ago. The learner's current forward pass provides fresh logits.
    """

    def __init__(self, delay: int):
        self.delay = delay
        self.buffer: deque[Batch] = deque(maxlen=delay + 1)

    def push(self, batch: Batch):
        self.buffer.append(batch.to('cpu'))

    def ready(self) -> bool:
        return len(self.buffer) > self.delay

    def get_stale(self, device) -> Batch:
        """Return the oldest batch in the queue (= D steps ago), on device."""
        return self.buffer[0].to(device)


# ── Gradient Diagnostics ─────────────────────────────────────────────────────


def compute_gradient_cosines(model, task, batch, loss_fn, method_logits_fn, device) -> dict[str, float]:
    """Cosine similarity of method gradient to CE oracle gradient.

    method_logits_fn: the logits function the method uses during training
    (compute_logits for RL methods, compute_logits_oracle for CE).
    """
    def flat_grad(logits_fn, compute_loss):
        model.zero_grad()
        logits = logits_fn(model, batch)
        loss = compute_loss(logits, batch)
        loss.backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    g_method = flat_grad(method_logits_fn, lambda l, b: loss_fn(l, b)[0])
    g_ce = flat_grad(task.compute_logits_oracle, lambda l, b: F.cross_entropy(
        l.reshape(-1, l.size(-1)), b.labels.reshape(-1)))

    cos = F.cosine_similarity
    result = {
        'cos_method_ce': cos(g_method.unsqueeze(0), g_ce.unsqueeze(0)).item(),
        'grad_norm': g_method.norm().item(),
    }
    model.zero_grad()
    return result


# ── Training Loop ────────────────────────────────────────────────────────────


def _use_autocast(config) -> bool:
    return config.task == 'lm_bandit'


def train_one_seed(task, loss_fn, model, config, seed, device) -> list[dict]:
    """Run one training seed. Returns list of metric dicts at eval points."""
    torch.manual_seed(seed)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    autocast = _use_autocast(config)

    # Compute difficulty terciles from the initial pre-trained model, before training
    task.compute_difficulty(model, device)

    # Determine group_size for DAPO
    group_size = config.dapo_group_size if config.method == 'DAPO' else 1
    assert config.batch_size % group_size == 0, \
        f'batch_size ({config.batch_size}) must be divisible by group_size ({group_size})'

    # Initialize experience queue: fill with D batches from the initial model
    queue = ExperienceQueue(config.delay)
    for _ in range(config.delay):
        init_batch = task.sample_batch(model, config.batch_size, device,
                                       group_size=group_size)
        queue.push(init_batch)

    results = []
    for step in range(config.num_steps):
        # Sample fresh batch with current model (acts as actor)
        fresh_batch = task.sample_batch(model, config.batch_size, device,
                                        group_size=group_size)
        queue.push(fresh_batch)
        batch = queue.get_stale(device)

        # Kondo screens samples before the forward pass — this is where it saves compute
        if config.method == 'Kondo':
            batch = batch.select(loss_fn.screen(batch))

        # Learner forward pass
        model.train()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
            # CE uses oracle logits (ground-truth prefix for sequential tasks)
            if config.method == 'CE':
                logits = task.compute_logits_oracle(model, batch)
            else:
                logits = task.compute_logits(model, batch)
            loss, metrics = loss_fn(logits, batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % config.eval_every == 0:
            model.eval()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                eval_metrics = task.evaluate(model, device)
            row = {'step': step, 'loss': loss.item(), **metrics, **eval_metrics}

            if config.diagnostics and step % (config.eval_every * 5) == 0:
                model.train()
                logits_fn = task.compute_logits_oracle if config.method == 'CE' else task.compute_logits
                row.update(compute_gradient_cosines(
                    model, task, batch, loss_fn, logits_fn, device))
                model.eval()

            results.append(row)
            if config.verbose and step % (config.eval_every * 10) == 0:
                print(f'  step {step:5d}  test_error={eval_metrics["test_error"]:.4f}'
                      f'  loss={loss.item():.4f}')

    return results


# ── Config and CLI ───────────────────────────────────────────────────────────


TASKS = {
    'mnist': lambda c: MNISTBandit(),
    'token_reversal': lambda c: TokenReversal(
        vocab_size=c.vocab_size, seq_len=c.seq_len),
    'lm_bandit': lambda c: LMBandit(
        model_name=c.model_name, context_len=c.context_len,
        kl_weight=c.kl_weight),
}

LOSSES = {
    'CE': lambda c: L.CELoss(),
    'REINFORCE': lambda c: L.REINFORCELoss(baseline=c.baseline),
    'PG': lambda c: L.PGLoss(baseline=c.baseline, iw_cap=c.iw_cap),
    'DG': lambda c: L.DGLoss(eta=c.eta, baseline=c.baseline),
    'Kondo': lambda c: L.KondoLoss(eta=c.eta, keep_ratio=c.kondo_keep, baseline=c.baseline),
    'LogGrowth': lambda c: L.LogGrowthLoss(baseline=c.baseline),
    'DAPO': lambda c: L.DAPOLoss(clip_low=c.clip_low, clip_high=c.clip_high),
    'PMDMean': lambda c: L.PMDMeanLoss(tau=c.eta),
}

MODEL_BUILDERS = {
    'mnist': lambda c, task: task.make_model(hidden=c.hidden),
    'token_reversal': lambda c, task: task.make_model(
        d_model=c.d_model, nhead=c.nhead, num_layers=c.num_layers),
    'lm_bandit': lambda c, task: task.make_model(),
}


@dataclass
class Config:
    task: str = 'mnist'
    method: str = 'DG'
    delay: int = 0
    num_steps: int = 1_000
    batch_size: int = 100
    lr: float = 1e-3
    eval_every: int = 20
    num_seeds: int = 5
    seed: int = 0
    baseline: str = 'expected'
    diagnostics: bool = False
    verbose: bool = True
    output: str = 'results.csv'
    sweep: bool = False
    # MLP
    hidden: int = 50
    # DG / Kondo / PMDMean
    eta: float = 1.0
    # PG
    iw_cap: float = 10.0
    # Kondo
    kondo_keep: float = 0.5
    # DAPO
    clip_low: float = 0.2
    clip_high: float = 0.28
    dapo_group_size: int = 4
    # Token reversal
    vocab_size: int = 2
    seq_len: int = 10
    d_model: int = 64
    nhead: int = 2
    num_layers: int = 2
    # LM bandit
    model_name: str = 'distilgpt2'
    context_len: int = 128
    kl_weight: float = 0.0


def run_config(config: Config) -> pd.DataFrame:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = TASKS[config.task](config)

    all_rows = []
    for i in range(config.num_seeds):
        seed = config.seed + i
        if config.verbose:
            print(f'{config.method} delay={config.delay} seed={seed}')
        torch.manual_seed(seed)
        loss_fn = LOSSES[config.method](config)
        model = MODEL_BUILDERS[config.task](config, task)
        rows = train_one_seed(task, loss_fn, model, config, seed, device)
        for r in rows:
            r.update({'seed': seed, 'method': config.method, 'delay': config.delay})
        all_rows.extend(rows)
        # Write after each seed so completed work survives crashes
        pd.DataFrame(all_rows).to_csv(config.output, index=False)

    return pd.DataFrame(all_rows)


def run_sweep(config: Config) -> pd.DataFrame:
    dfs = []
    for method in ['REINFORCE', 'PG', 'DG']:
        for delay in [0, 1, 3, 10, 30, 100]:
            cfg = dataclasses.replace(config, method=method, delay=delay)
            dfs.append(run_config(cfg))
    return pd.concat(dfs, ignore_index=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

TYPE_MAP = {int: int, float: float, str: str, bool: bool}


def main():
    parser = argparse.ArgumentParser(description='Delightful Policy Gradient')
    for f in dataclasses.fields(Config):
        ty = TYPE_MAP[f.type] if f.type in TYPE_MAP else str
        if f.type is bool:
            parser.add_argument(f'--{f.name}', type=lambda x: x.lower() == 'true',
                                default=f.default, metavar='BOOL')
        else:
            parser.add_argument(f'--{f.name}', type=ty, default=f.default)
    args = parser.parse_args()
    config = Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})

    t0 = time.time()
    df = run_sweep(config) if config.sweep else run_config(config)
    df.to_csv(config.output, index=False)
    print(f'Saved {len(df)} rows to {config.output} ({time.time() - t0:.1f}s)')


if __name__ == '__main__':
    main()
