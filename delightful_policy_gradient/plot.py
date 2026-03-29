"""Plot experiment results from CSV files."""

import argparse

import pandas as pd
import plotnine as gg

gg.theme_set(gg.theme_bw(base_size=14))


def plot_delay_sweep(df: pd.DataFrame, output: str | None = None):
    """Test error vs step, faceted by delay, colored by method."""
    mean_df = (df.groupby(['method', 'step', 'delay'])['test_error']
               .agg(['mean', 'sem']).reset_index())

    p = (gg.ggplot(mean_df)
         + gg.aes(x='step', y='mean', colour='method', fill='method')
         + gg.geom_ribbon(gg.aes(ymin='mean-sem', ymax='mean+sem'), alpha=0.2, size=0)
         + gg.geom_line()
         + gg.facet_wrap('delay', labeller='label_both', nrow=1)
         + gg.theme(figure_size=(16, 4))
         + gg.scale_y_log10()
         + gg.ylab('test error'))

    if output:
        p.save(output, dpi=150)
        print(f'Saved to {output}')
    else:
        print(p)


def plot_final_vs_delay(df: pd.DataFrame, output: str | None = None):
    """Final test error vs delay for each method."""
    final_df = df[df.step == df.step.max()]
    mean_df = (final_df.groupby(['method', 'delay'])['test_error']
               .agg(['mean', 'sem']).reset_index())

    p = (gg.ggplot(mean_df)
         + gg.aes(x='delay', y='mean', colour='method')
         + gg.geom_errorbar(gg.aes(ymin='mean-sem', ymax='mean+sem'), alpha=0.5)
         + gg.geom_line()
         + gg.geom_point()
         + gg.theme(figure_size=(6, 4))
         + gg.scale_x_log10()
         + gg.scale_y_log10()
         + gg.ylab('test error at final step'))

    if output:
        p.save(output, dpi=150)
        print(f'Saved to {output}')
    else:
        print(p)


def main():
    parser = argparse.ArgumentParser(description='Plot DPG results')
    parser.add_argument('csv', help='Path to results CSV')
    parser.add_argument('--output', '-o', help='Save figure to file')
    parser.add_argument('--kind', default='sweep',
                        choices=['sweep', 'final'],
                        help='sweep: error vs step faceted by delay; '
                             'final: final error vs delay')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.kind == 'sweep':
        plot_delay_sweep(df, args.output)
    else:
        plot_final_vs_delay(df, args.output)


if __name__ == '__main__':
    main()
