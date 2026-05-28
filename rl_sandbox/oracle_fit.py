"""Oracle influence analysis: what does the optimal influence function look like?

For a batch of samples, computes:
  1. Per-sample gradient vectors (how each sample pushes model parameters)
  2. CE oracle gradient (the supervised direction we'd follow with labels)
  3. Oracle projection per sample (dot product of sample gradient with oracle)
  4. Delight per sample (advantage x surprisal, what DG uses)

If DG's gate correctly identifies important samples, oracle projection
and delight should correlate. If they don't, the optimal influence function
depends on something other than advantage x surprisal.

Usage:
  python -m rl_sandbox.oracle_fit --task mnist --train_steps 0,100,200
  python -m rl_sandbox.oracle_fit --task token_reversal
  python -m rl_sandbox.oracle_fit --task lm_bandit --model_name distilgpt2 --batch_size 16
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import torch
import torch.nn.functional as F
import pandas as pd
import plotnine as gg

from .losses import REINFORCELoss, gather_log_probs, compute_baseline
from .train import Config, TASKS, MODEL_BUILDERS, validate_config

gg.theme_set(gg.theme_bw(base_size=14))


def flat_parameter_gradients(model: torch.nn.Module) -> torch.Tensor:
    """Flatten gradients, filling unused parameters with zeros."""
    flat = []
    for parameter in model.parameters():
        if parameter.grad is None:
            flat.append(torch.zeros_like(parameter).reshape(-1))
        else:
            flat.append(parameter.grad.detach().reshape(-1))
    if not flat:
        return torch.zeros(1)
    return torch.cat(flat)


def per_sample_grads(model, logits_fn, batch, device):
    """Compute per-sample gradient of log pi(action|obs). Returns [B, P] tensor.

    Each row is the gradient of one sample's log-probability contribution.
    Uses a loop over samples for clarity. Fine for diagnostics.
    """
    P = sum(p.numel() for p in model.parameters())
    B = batch.obs.shape[0]
    grads = torch.zeros(B, P, device=device)

    for i in range(B):
        single = batch.select(torch.tensor([i], device=device))
        model.zero_grad(set_to_none=True)
        logits = logits_fn(model, single)
        log_probs = F.log_softmax(logits, dim=-1)
        logp_a = gather_log_probs(log_probs, single.actions)
        logp_a.sum().backward()
        grads[i] = flat_parameter_gradients(model)

    model.zero_grad(set_to_none=True)
    return grads


def advantage_from_batch(logits, batch):
    """Match the training loss advantage convention for diagnostics."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    logp_a = gather_log_probs(log_probs, batch.actions)

    if batch.actor_expected_reward is not None:
        baseline = batch.actor_expected_reward
    else:
        baseline = compute_baseline('expected', probs)

    reward = batch.rewards
    while reward.dim() < baseline.dim():
        reward = reward.unsqueeze(-1)
    advantage = reward - baseline
    while advantage.dim() < logp_a.dim():
        advantage = advantage.unsqueeze(-1)
    return logp_a, advantage


def analyze_batch(model, task, batch, device):
    """Compute oracle projections, delight, and correlations for one batch.

    Returns a DataFrame with one row per sample: delight, oracle_proj,
    advantage, surprisal, reward, and the DG gate value.
    """
    was_training = model.training
    model.eval()

    # Per-sample PG gradients: gradient of log pi(a_i|x_i)
    pg_grads = per_sample_grads(model, task.compute_logits, batch, device)

    # CE oracle gradient: mean gradient of log pi(y*|x) over the batch
    model.zero_grad(set_to_none=True)
    oracle_logits = task.compute_logits_oracle(model, batch)
    ce_loss = F.cross_entropy(
        oracle_logits.reshape(-1, oracle_logits.size(-1)),
        batch.labels.reshape(-1))
    ce_loss.backward()
    oracle_grad = flat_parameter_gradients(model)
    model.zero_grad(set_to_none=True)

    # Oracle projection: how much each sample's gradient aligns with CE
    projections = pg_grads @ oracle_grad  # [B]

    # Delight: advantage x surprisal (what DG uses to gate)
    with torch.no_grad():
        logits = task.compute_logits(model, batch)
        logp_a, advantage = advantage_from_batch(logits, batch)
        surprisal = -logp_a
        # For sequential tasks, reduce to per-sequence
        if surprisal.dim() > 1:
            surprisal = surprisal.mean(dim=-1)
            delight = (advantage * (-logp_a)).mean(dim=-1)
            advantage_report = advantage.mean(dim=-1)
        else:
            delight = advantage * surprisal
            advantage_report = advantage
        gate = torch.sigmoid(delight)

    df = pd.DataFrame({
        'oracle_proj': projections.detach().cpu().numpy(),
        'delight': delight.cpu().numpy(),
        'advantage': advantage_report.cpu().numpy(),
        'surprisal': surprisal.cpu().numpy(),
        'reward': batch.rewards.cpu().numpy(),
        'gate': gate.cpu().numpy(),
    })
    model.train(was_training)
    return df


def run_analysis(config, train_steps_list=(0, 100, 500)):
    """Run oracle analysis at multiple training stages."""
    validate_config(config)
    if not train_steps_list:
        raise ValueError('train_steps_list must contain at least one step')
    if min(train_steps_list) < 0:
        raise ValueError('train_steps_list must be non-negative')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    task = TASKS[config.task](config)
    torch.manual_seed(config.seed)
    model = MODEL_BUILDERS[config.task](config, task).to(device)
    task.compute_difficulty(model, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    autocast = config.task == 'lm_bandit' and device.type == 'cuda'
    reinforce = REINFORCELoss()

    all_dfs = []
    next_analysis = iter(sorted(train_steps_list))
    next_step = next(next_analysis, None)

    for step in range(max(train_steps_list) + 1):
        if step == next_step:
            # Analyze current model state
            batch = task.sample_batch(model, config.batch_size, device)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
                df = analyze_batch(model, task, batch, device)
            df['train_step'] = step
            corr = df['oracle_proj'].corr(df['delight'])
            print(f'step={step:4d}  corr(oracle_proj, delight)={corr:.3f}'
                  f'  mean_reward={df["reward"].mean():.3f}')
            all_dfs.append(df)
            next_step = next(next_analysis, None)

        if step == max(train_steps_list):
            break

        # Train one step
        batch = task.sample_batch(model, config.batch_size, device)
        model.eval()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
            logits = task.compute_logits(model, batch)
            loss, _ = reinforce(logits, batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return pd.concat(all_dfs, ignore_index=True)


def plot_results(df, output_prefix='results/oracle_fit'):
    """Scatter plots of oracle projection vs delight at each training stage."""
    p = (gg.ggplot(df)
         + gg.aes(x='delight', y='oracle_proj', color='reward')
         + gg.geom_point(alpha=0.5, size=1.5)
         + gg.facet_wrap('train_step', labeller='label_both', nrow=1)
         + gg.theme(figure_size=(5 * df['train_step'].nunique(), 4))
         + gg.xlab('delight (advantage x surprisal)')
         + gg.ylab('oracle projection (gradient dot CE oracle)')
         + gg.ggtitle('Does delight predict which samples help reach the CE oracle?'))
    p.save(f'{output_prefix}.png', dpi=150)
    print(f'Saved {output_prefix}.png')

    # Also plot gate vs oracle projection
    p2 = (gg.ggplot(df)
          + gg.aes(x='gate', y='oracle_proj', color='reward')
          + gg.geom_point(alpha=0.5, size=1.5)
          + gg.facet_wrap('train_step', labeller='label_both', nrow=1)
          + gg.theme(figure_size=(5 * df['train_step'].nunique(), 4))
          + gg.xlab('DG gate value sigmoid(delight)')
          + gg.ylab('oracle projection')
          + gg.ggtitle('Does the DG gate predict oracle-aligned samples?'))
    p2.save(f'{output_prefix}_gate.png', dpi=150)
    print(f'Saved {output_prefix}_gate.png')


def main():
    parser = argparse.ArgumentParser(description='Oracle influence analysis')
    parser.add_argument('--task', default='mnist')
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=50)
    parser.add_argument('--model_name', default='distilgpt2')
    parser.add_argument('--context_len', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=5)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--train_steps', default='0,100,500',
                        help='Comma-separated training steps to analyze at')
    parser.add_argument('--output', default='results/oracle_fit')
    args = parser.parse_args()

    config = Config(
        task=args.task, batch_size=args.batch_size, lr=args.lr,
        seed=args.seed, hidden=args.hidden, model_name=args.model_name,
        context_len=args.context_len, vocab_size=args.vocab_size,
        seq_len=args.seq_len, d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers)

    steps = [int(s.strip()) for s in args.train_steps.split(',') if s.strip()]
    df = run_analysis(config, train_steps_list=steps)
    output_parent = Path(args.output).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f'{args.output}.csv', index=False)
    plot_results(df, output_prefix=args.output)


if __name__ == '__main__':
    main()
