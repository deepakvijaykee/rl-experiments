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


def style_axis(ax, ylabel: str, title: str | None = None):
    ax.set_xlabel('step')
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(axis='y', alpha=0.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


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


def main():
    plot_influence()
    plot_noise()
    plot_replay()
    plot_partial_credit()
    plot_dense_correction()
    plot_entropy()
    print(f'Saved evidence figures to {OUT}')


if __name__ == '__main__':
    main()
