"""Generate compact learning-curve evidence plots for the README."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'rl_sandbox' / 'analysis' / 'figures'

COLORS = {
    'DG': '#4C78A8',
    'DG delay4': '#4C78A8',
    'DGEntropyGuard': '#72B7B2',
    'GRPO': '#F58518',
    'TPO': '#54A24B',
    'CE': '#6B6ECF',
    'ASPO': '#E45756',
    'R2VPO': '#B279A2',
    'UncertaintyDG': '#9D755D',
    'RewardVarianceDG': '#FF9DA6',
    'ReplayDG cap32': '#F58518',
    'FreshDG cap5': '#54A24B',
    'FreshDG cap32': '#B279A2',
    'FreshDG decay0.5': '#E45756',
    'DGToken': '#72B7B2',
    'TPOToken': '#54A24B',
    'GRPOToken': '#F58518',
    'SelfDistillDG': '#54A24B',
    'SCOPELite': '#E45756',
}


def read_run(path: str, label: str) -> pd.DataFrame:
    csv_path = ROOT / path
    if not csv_path.exists():
        raise FileNotFoundError(f'{csv_path} is missing; run the sweep first')
    df = pd.read_csv(csv_path)
    df['label'] = label
    return df


def combine(runs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.concat([read_run(path, label) for path, label in runs],
                     ignore_index=True)


def step_summary(df: pd.DataFrame, value: str) -> pd.DataFrame:
    data = df.dropna(subset=[value]).copy()
    summary = (data.groupby(['label', 'step'], sort=False)[value]
               .agg(['mean', 'sem']).fillna(0.0).reset_index())
    return summary


def final_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby(['label', 'seed']).step.idxmax()].copy()


def final_summary(df: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = final_rows(df).dropna(subset=[value])
    return (rows.groupby('label', sort=False)[value]
            .agg(['mean', 'sem']).fillna(0.0).reset_index())


def mean_by_seed(df: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = df.dropna(subset=[value])
    return (rows.groupby(['label', 'seed'], sort=False)[value]
            .mean().reset_index())


def metric_summary(rows: pd.DataFrame, value: str) -> pd.DataFrame:
    return (rows.groupby('label', sort=False)[value]
            .agg(['mean', 'sem']).fillna(0.0).reset_index())


def first_crossing_rows(
    df: pd.DataFrame,
    value: str,
    threshold: float,
) -> pd.DataFrame:
    records = []
    for (label, seed), rows in df.dropna(subset=[value]).groupby(['label', 'seed']):
        hit = rows.loc[rows[value] <= threshold, 'step']
        records.append({
            'label': label,
            'seed': seed,
            'first_step': hit.iloc[0] if len(hit) else float('nan'),
        })
    return pd.DataFrame(records)


def style_axis(ax, ylabel: str, title: str | None = None):
    ax.set_xlabel('step')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis='y', alpha=0.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def label_points(ax, rows: pd.DataFrame, x: str, y: str):
    for _, row in rows.iterrows():
        ax.annotate(row['label'], (row[x], row[y]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)


def draw_frontier(
    ax,
    rows: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    title: str,
):
    for _, row in rows.iterrows():
        color = COLORS.get(row['label'])
        ax.errorbar(row[x], row[y],
                    xerr=row.get(f'{x}_sem', 0.0),
                    yerr=row.get(f'{y}_sem', 0.0),
                    fmt='o', markersize=6, capsize=3,
                    color=color, ecolor=color)
    label_points(ax, rows, x, y)
    style_axis(ax, ylabel, title)
    ax.set_xlabel(xlabel)


def draw_curves(
    ax,
    df: pd.DataFrame,
    value: str,
    ylabel: str,
    title: str | None = None,
    labels: list[str] | None = None,
):
    labels = labels or list(dict.fromkeys(df['label']))
    summary = step_summary(df, value)

    for label in labels:
        rows = summary[summary['label'] == label]
        if rows.empty:
            continue
        color = COLORS.get(label)
        x = rows['step'].to_numpy()
        y = rows['mean'].to_numpy()
        err = rows['sem'].to_numpy()
        ax.plot(x, y, label=label, linewidth=2.2, color=color)
        ax.fill_between(x, y - err, y + err, alpha=0.16, color=color)

    style_axis(ax, ylabel, title)


def add_figure_legend(fig, ax, ncol: int):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=ncol,
               loc='lower center', bbox_to_anchor=(0.5, 0.01))


def save(fig, name: str, bottom: float = 0.0):
    OUT.mkdir(parents=True, exist_ok=True)
    if bottom:
        fig.tight_layout(rect=(0, bottom, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)


def plot_influence():
    data = combine([
        ('results/dg_token.csv', 'DG'),
        ('results/grpo_token.csv', 'GRPO'),
        ('results/tpo_token.csv', 'TPO'),
    ])
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    draw_curves(ax, data, 'test_error', 'test error',
                'Clean Token Reversal')
    ax.set_ylim(0.15, 0.75)
    ax.legend(frameon=False, ncol=3, loc='upper right')
    save(fig, 'influence.png')


def plot_noise():
    data = combine([
        ('results/noise_dg_false_positive_rare.csv', 'DG'),
        ('results/noise_uncertaintydg_false_positive_rare.csv',
         'UncertaintyDG'),
        ('results/noise_rewardvariancedg_false_positive_rare.csv',
         'RewardVarianceDG'),
        ('results/noise_aspo_false_positive_rare.csv', 'ASPO'),
        ('results/noise_r2vpo_false_positive_rare.csv', 'R2VPO'),
    ])
    labels = ['DG', 'UncertaintyDG', 'RewardVarianceDG', 'ASPO', 'R2VPO']
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharex=True)
    draw_curves(axes[0], data, 'test_error', 'test error',
                'False-Positive Reward Noise', labels)
    draw_curves(axes[1], data, 'entropy', 'entropy',
                'Collapse Pressure', labels)
    axes[0].set_ylim(0.3, 0.75)
    add_figure_legend(fig, axes[0], ncol=5)
    save(fig, 'reward_noise.png', bottom=0.16)


def plot_replay():
    data = combine([
        ('results/replay_dg_delay4.csv', 'DG delay4'),
        ('results/replay_freshdg_cap5_delay4.csv', 'FreshDG cap5'),
        ('results/replay_replaydg_delay4.csv', 'ReplayDG cap32'),
        ('results/replay_freshdg_delay4.csv', 'FreshDG cap32'),
        ('results/replay_freshdg_decay05_delay4.csv', 'FreshDG decay0.5'),
    ])
    labels = ['DG delay4', 'FreshDG cap5', 'ReplayDG cap32',
              'FreshDG cap32', 'FreshDG decay0.5']
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharex=True)
    draw_curves(axes[0], data, 'test_error', 'test error',
                'Replay Under Policy Drift', labels)
    draw_curves(axes[1], data, 'replay_age', 'sample age',
                'Replay Age Actually Used', labels)
    axes[0].set_ylim(0.45, 1.03)
    add_figure_legend(fig, axes[0], ncol=5)
    save(fig, 'replay.png', bottom=0.16)


def plot_partial_credit():
    data = combine([
        ('results/masked_axis_ce.csv', 'CE'),
        ('results/masked_axis_dg.csv', 'DG'),
        ('results/masked_axis_dgtoken.csv', 'DGToken'),
        ('results/masked_axis_tpotoken.csv', 'TPOToken'),
        ('results/masked_axis_grpotoken.csv', 'GRPOToken'),
    ])
    labels = ['CE', 'DG', 'DGToken', 'TPOToken', 'GRPOToken']
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharex=True)
    draw_curves(axes[0], data, 'test_error', 'scored error',
                'Masked Reversal Scored Tokens', labels)
    draw_curves(axes[1], data, 'test_error_unscored', 'unscored error',
                'Unscored Positions', labels)
    axes[0].set_ylim(-0.02, 0.75)
    axes[1].set_ylim(0.2, 0.75)
    add_figure_legend(fig, axes[0], ncol=5)
    save(fig, 'partial_credit.png', bottom=0.16)


def plot_dense_correction():
    data = combine([
        ('results/chain_ce_1500.csv', 'CE'),
        ('results/chain_selfdistilldg_1500.csv', 'SelfDistillDG'),
        ('results/chain_scopelite_1500.csv', 'SCOPELite'),
    ])
    labels = ['CE', 'SelfDistillDG', 'SCOPELite']
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharex=True)
    draw_curves(axes[0], data, 'test_error', 'exact-match error',
                'Reward-Chain Reversal', labels)
    draw_curves(axes[1], data, 'chain_reward', 'chain reward',
                'Checkpoint Credit', labels)
    axes[0].set_ylim(-0.02, 1.05)
    axes[1].set_ylim(-0.02, 1.05)
    add_figure_legend(fig, axes[0], ncol=3)
    save(fig, 'dense_correction.png', bottom=0.16)


def plot_entropy():
    data = combine([
        ('results/entropy_dg.csv', 'DG'),
        ('results/entropy_dgentropyguard.csv', 'DGEntropyGuard'),
        ('results/entropy_aspo.csv', 'ASPO'),
        ('results/entropy_r2vpo.csv', 'R2VPO'),
        ('results/entropy_grpo.csv', 'GRPO'),
        ('results/entropy_tpo.csv', 'TPO'),
    ])
    labels = ['DG', 'DGEntropyGuard', 'ASPO', 'R2VPO', 'GRPO', 'TPO']
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7), sharex=True)
    draw_curves(axes[0], data, 'test_error', 'test error',
                'Accuracy vs Entropy Collapse', labels)
    draw_curves(axes[1], data, 'entropy', 'entropy',
                'Policy Entropy', labels)
    axes[0].set_ylim(0.15, 0.75)
    axes[1].set_ylim(-0.02, 1.15)
    add_figure_legend(fig, axes[0], ncol=6)
    save(fig, 'entropy.png', bottom=0.16)


def plot_utility_tradeoffs():
    entropy = combine([
        ('results/entropy_dg.csv', 'DG'),
        ('results/entropy_dgentropyguard.csv', 'DGEntropyGuard'),
        ('results/entropy_aspo.csv', 'ASPO'),
        ('results/entropy_r2vpo.csv', 'R2VPO'),
        ('results/entropy_grpo.csv', 'GRPO'),
        ('results/entropy_tpo.csv', 'TPO'),
    ])
    entropy_error = final_summary(entropy, 'test_error')
    entropy_level = final_summary(entropy, 'entropy')
    entropy_frontier = entropy_level.merge(entropy_error, on='label',
                                           suffixes=('_entropy', '_error'))
    entropy_frontier = entropy_frontier.rename(columns={
        'mean_entropy': 'entropy',
        'sem_entropy': 'entropy_sem',
        'mean_error': 'error',
        'sem_error': 'error_sem',
    })

    replay = combine([
        ('results/replay_freshdg_cap5_delay4.csv', 'FreshDG cap5'),
        ('results/replay_replaydg_delay4.csv', 'ReplayDG cap32'),
        ('results/replay_freshdg_delay4.csv', 'FreshDG cap32'),
        ('results/replay_freshdg_decay05_delay4.csv', 'FreshDG decay0.5'),
    ])
    replay_error = final_summary(replay, 'test_error')
    replay_age = metric_summary(mean_by_seed(replay, 'replay_age'), 'replay_age')
    replay_frontier = replay_age.merge(replay_error, on='label',
                                       suffixes=('_age', '_error'))
    replay_frontier = replay_frontier.rename(columns={
        'mean_age': 'age',
        'sem_age': 'age_sem',
        'mean_error': 'error',
        'sem_error': 'error_sem',
    })

    partial = combine([
        ('results/masked_axis_ce.csv', 'CE'),
        ('results/masked_axis_dg.csv', 'DG'),
        ('results/masked_axis_dgtoken.csv', 'DGToken'),
        ('results/masked_axis_tpotoken.csv', 'TPOToken'),
        ('results/masked_axis_grpotoken.csv', 'GRPOToken'),
    ])
    partial_scored = final_summary(partial, 'test_error')
    partial_unscored = final_summary(partial, 'test_error_unscored')
    partial_frontier = partial_unscored.merge(partial_scored, on='label',
                                             suffixes=('_unscored', '_scored'))
    partial_frontier = partial_frontier.rename(columns={
        'mean_unscored': 'unscored',
        'sem_unscored': 'unscored_sem',
        'mean_scored': 'scored',
        'sem_scored': 'scored_sem',
    })

    dense = combine([
        ('results/chain_ce_1500.csv', 'CE'),
        ('results/chain_selfdistilldg_1500.csv', 'SelfDistillDG'),
        ('results/chain_scopelite_1500.csv', 'SCOPELite'),
    ])
    hit = first_crossing_rows(dense, 'test_error', 0.0)
    hit_summary = metric_summary(hit, 'first_step')
    reward_auc = metric_summary(mean_by_seed(dense, 'chain_reward'), 'chain_reward')
    dense_frontier = hit_summary.merge(reward_auc, on='label',
                                       suffixes=('_hit', '_reward'))
    dense_frontier = dense_frontier.rename(columns={
        'mean_hit': 'first_step',
        'sem_hit': 'first_step_sem',
        'mean_reward': 'chain_reward',
        'sem_reward': 'chain_reward_sem',
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))
    draw_frontier(axes[0, 0], entropy_frontier, 'entropy', 'error',
                  'final entropy', 'final test error',
                  'Support/Accuracy Frontier')
    draw_frontier(axes[0, 1], replay_frontier, 'age', 'error',
                  'mean replay age', 'final test error',
                  'Staleness Frontier')
    draw_frontier(axes[1, 0], partial_frontier, 'unscored', 'scored',
                  'unscored error', 'scored error',
                  'Credit Specificity')
    draw_frontier(axes[1, 1], dense_frontier, 'first_step', 'chain_reward',
                  'first zero-error step', 'mean chain reward',
                  'Dense-Correction Efficiency')
    save(fig, 'utility_tradeoffs.png')


def plot_entropy_buckets():
    data = combine([
        ('results/entropy_dg.csv', 'DG'),
        ('results/entropy_dgentropyguard.csv', 'DGEntropyGuard'),
        ('results/entropy_grpo.csv', 'GRPO'),
        ('results/entropy_tpo.csv', 'TPO'),
        ('results/entropy_aspo.csv', 'ASPO'),
        ('results/entropy_r2vpo.csv', 'R2VPO'),
    ])
    labels = ['DG', 'DGEntropyGuard', 'GRPO', 'TPO', 'ASPO', 'R2VPO']
    bucket_names = ['low', 'mid', 'high']
    families = {
        'surprisal': [f'entropy_drop_surprisal_{name}'
                      for name in bucket_names],
        'delight': [f'entropy_drop_delight_{name}' for name in bucket_names],
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), sharey=True)
    for ax, (family, cols) in zip(axes, families.items()):
        for label in labels:
            rows = data[data['label'] == label]
            values = [rows[col].dropna().mean() for col in cols]
            ax.plot(bucket_names, values, marker='o', linewidth=2.0,
                    color=COLORS.get(label), label=label)
        ax.axhline(0.0, color='#777777', linewidth=0.8, alpha=0.5)
        ax.set_title(f'Entropy Drop by {family.title()} Bucket')
        ax.set_xlabel('bucket')
        ax.grid(axis='y', alpha=0.18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('mean entropy drop')
    add_figure_legend(fig, axes[0], ncol=6)
    save(fig, 'entropy_buckets.png', bottom=0.17)


def main():
    plot_influence()
    plot_noise()
    plot_replay()
    plot_partial_credit()
    plot_dense_correction()
    plot_entropy()
    plot_utility_tradeoffs()
    plot_entropy_buckets()
    print(f'Saved evidence figures to {OUT}')


if __name__ == '__main__':
    main()
