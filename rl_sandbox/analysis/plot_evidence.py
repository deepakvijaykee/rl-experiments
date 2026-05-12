"""Generate compact evidence plots for the top-level README."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'rl_sandbox' / 'analysis' / 'figures'


def final_by_seed(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / path)
    final = df.loc[df.groupby('seed').step.idxmax()].copy()
    final['label'] = label
    return final


def mean_sem(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return (df.groupby('label', sort=False)[value]
            .agg(['mean', 'sem']).fillna(0.0).reset_index())


def bar(ax, summary: pd.DataFrame, value_label: str, color: str):
    ax.bar(summary['label'], summary['mean'], yerr=summary['sem'],
           color=color, alpha=0.86, capsize=3)
    ax.set_ylabel(value_label)
    ax.tick_params(axis='x', rotation=25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)


def plot_influence():
    data = pd.concat([
        final_by_seed('results/dg_token.csv', 'DG'),
        final_by_seed('results/grpo_token.csv', 'GRPO'),
        final_by_seed('results/tpo_token.csv', 'TPO'),
    ])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bar(ax, mean_sem(data, 'test_error'), 'final test error', '#4C78A8')
    ax.set_title('Clean Token Reversal')
    save(fig, 'influence.png')


def plot_noise():
    data = pd.concat([
        final_by_seed('results/noise_dg_false_positive_rare.csv', 'DG'),
        final_by_seed('results/noise_uncertaintydg_false_positive_rare.csv',
                      'UncertaintyDG'),
        final_by_seed('results/noise_rewardvariancedg_false_positive_rare.csv',
                      'RewardVarianceDG'),
        final_by_seed('results/noise_aspo_false_positive_rare.csv', 'ASPO'),
        final_by_seed('results/noise_r2vpo_false_positive_rare.csv', 'R2VPO'),
    ])
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bar(ax, mean_sem(data, 'test_error'), 'final test error', '#F58518')
    ax.set_title('False-Positive Rare-Token Noise')
    save(fig, 'reward_noise.png')


def plot_replay():
    data = pd.concat([
        final_by_seed('results/replay_dg_delay4.csv', 'DG delay4'),
        final_by_seed('results/replay_freshdg_cap5_delay4.csv', 'FreshDG cap5'),
        final_by_seed('results/replay_freshdg_delay4.csv', 'FreshDG cap32'),
        final_by_seed('results/replay_freshdg_decay05_delay4.csv',
                      'FreshDG cap32 decay0.5'),
    ])
    fig, ax = plt.subplots(figsize=(6.1, 3.2))
    bar(ax, mean_sem(data, 'test_error'), 'final test error', '#54A24B')
    ax.set_title('Replay Freshness Under Delay')
    save(fig, 'replay.png')


def plot_partial_credit():
    runs = [
        final_by_seed('results/masked_axis_ce.csv', 'CE'),
        final_by_seed('results/masked_axis_dg.csv', 'DG'),
        final_by_seed('results/masked_axis_dgtoken.csv', 'DGToken'),
        final_by_seed('results/masked_axis_tpotoken.csv', 'TPOToken'),
        final_by_seed('results/masked_axis_grpotoken.csv', 'GRPOToken'),
    ]
    data = pd.concat(runs)
    scored = mean_sem(data, 'test_error')
    unscored = mean_sem(data, 'test_error_unscored')

    x = range(len(scored))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.bar([i - width / 2 for i in x], scored['mean'], width,
           yerr=scored['sem'], label='scored', color='#B279A2', capsize=3)
    ax.bar([i + width / 2 for i in x], unscored['mean'], width,
           yerr=unscored['sem'], label='unscored', color='#9D755D', capsize=3)
    ax.set_xticks(list(x), scored['label'], rotation=25)
    ax.set_ylabel('final test error')
    ax.set_title('Masked Reversal Credit')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    save(fig, 'partial_credit.png')


def plot_dense_correction():
    runs = [
        final_by_seed('results/chain_ce_1500.csv', 'CE'),
        final_by_seed('results/chain_selfdistilldg_1500.csv', 'SelfDistillDG'),
        final_by_seed('results/chain_scopelite_1500.csv', 'SCOPELite'),
    ]
    data = []
    for run in runs:
        for seed, rows in run.groupby('seed'):
            hit = rows.loc[rows['test_error'] <= 0.0, 'step']
            data.append({
                'label': rows['label'].iloc[0],
                'seed': seed,
                'first_zero_step': hit.iloc[0] if len(hit) else None,
            })
    summary = mean_sem(pd.DataFrame(data), 'first_zero_step')
    fig, ax = plt.subplots(figsize=(5.3, 3.2))
    bar(ax, summary, 'first zero-error step', '#E45756')
    ax.set_title('Reward-Chain Dense Correction')
    save(fig, 'dense_correction.png')


def plot_entropy():
    data = pd.concat([
        final_by_seed('results/entropy_dg.csv', 'DG'),
        final_by_seed('results/entropy_dgentropyguard.csv', 'DGEntropyGuard'),
        final_by_seed('results/entropy_aspo.csv', 'ASPO'),
        final_by_seed('results/entropy_r2vpo.csv', 'R2VPO'),
        final_by_seed('results/entropy_grpo.csv', 'GRPO'),
        final_by_seed('results/entropy_tpo.csv', 'TPO'),
    ])
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    bar(ax, mean_sem(data, 'entropy'), 'final entropy', '#72B7B2')
    ax.set_title('Entropy After Compact Training')
    save(fig, 'entropy.png')


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
