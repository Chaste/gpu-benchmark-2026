"""Generate publication-quality performance figures for the GPU-simulator study.

This replaces the earlier set of six standalone plots with two coherent
multi-panel figures, and refines the styling to journal standards:

    Figure 1 (fig-profiling):  runtime profiling, 2x2
        (A) proportion of runtime spent on mechanics      [line, log x]
        (B) proportion of runtime spent on CPU activities [line, log x]
        (C) composition of GPU runtime                    [stacked bar]
        (D) absolute runtime by stage                     [stacked area, linear x]

    Figure 2 (fig-speedup):    speedup, 1x2
        (A) overall speedup: actual vs maximum theoretical   [line, log x]
        (B) mechanics-step speedup, 2D and 3D                [line, log x]

Each figure is written as both a vector PDF (for submission) and a 300-dpi PNG
(for preview). Run as a script with no arguments::

    python graphs.py

Abbreviations (used in the stage labels of the profiling figure)
    D2H = Device to Host   (CSV columns prefixed 'DTH')
    H2D = Host to Device   (CSV columns prefixed 'HTD')
    "translation" is the data-wrangling step ('Wrangle' columns); "transfer" is
    the raw copy ('Transfer' columns).

Presentation choices (all easily reverted; see the CONFIG block):
  * Fonts use Arial/Helvetica where available, falling back to Liberation Sans
    (an Arial-metric-compatible substitute) and then DejaVu Sans.
  * Population, not box width, is the common x-axis throughout, so panels are
    directly comparable. Population is width**2 in 2D and width**3 in 3D; for the
    mechanics-proportion data it is (box size)**2.
  * A logarithmic x-axis is used for the line panels, because population spans
    four orders of magnitude; this both spreads the small-population points and
    renders the axis in scientific (10**n) notation, as the reviewer requested.
    The stacked-area panel keeps a linear x-axis (area is misleading on a log
    axis) with an explicit scientific-notation offset. The stacked-bar panel is
    categorical, so its populations are shown against a common 10**5 factor.
  * Both proportion panels are shown as percentages for consistency.
  * A colourblind-friendly palette (Okabe-Ito) is used, with 2D and 3D drawn in
    the same two colours across every panel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, ScalarFormatter

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
USE_LOG_X = True          # log x-axis for the line panels
SAVE_PDF = True           # write vector PDFs for submission
SAVE_PNG = True           # write 300-dpi PNGs for preview

# Okabe-Ito colourblind-safe palette
OKABE = {
    'orange': '#E69F00', 'sky': '#56B4E9', 'green': '#009E73',
    'yellow': '#F0E442', 'blue': '#0072B2', 'verm': '#D55E00',
    'purple': '#CC79A7', 'grey': '#999999',
}
C2D, C3D = OKABE['blue'], OKABE['verm']

# Consistent stage colours, shared between the stacked bar and stacked area.
# Keys are the full stage names; D2H = Device to Host, H2D = Host to Device.
STAGE_COLOURS = {
    'Simulation':                        OKABE['blue'],
    'CPU time':                          OKABE['grey'],
    'Device to host data translation':   OKABE['green'],
    'Device to host transfer':           OKABE['sky'],
    'Host to device transfer':           OKABE['orange'],
    'Host to device data translation':   OKABE['purple'],
}

# --------------------------------------------------------------------------- #
# Global style — deliberately not seaborn's default theme.
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'Liberation Sans',
                          'Nimbus Sans', 'DejaVu Sans'],
    # Match the maths glyphs (the 10**n labels) to the Arial-like body font.
    'mathtext.fontset':  'custom',
    'mathtext.rm':       'Liberation Sans',
    'mathtext.it':       'Liberation Sans:italic',
    'mathtext.bf':       'Liberation Sans:bold',
    'mathtext.sf':       'Liberation Sans',
    'mathtext.cal':      'Liberation Sans:italic',
    'mathtext.tt':       'Liberation Mono',
    'font.size':         10,
    'axes.titlesize':    11,
    'axes.labelsize':    11,
    'xtick.labelsize':   9,
    'ytick.labelsize':   9,
    'legend.fontsize':   8,
    'legend.frameon':    True,
    'legend.framealpha': 0.9,
    'legend.edgecolor':  '0.8',
    'lines.linewidth':   1.6,
    'lines.markersize':  4.0,
    'axes.linewidth':    0.8,
    'axes.grid':         True,
    'grid.color':        '0.85',
    'grid.linewidth':    0.6,
    'axes.axisbelow':    True,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

ERRORBAR_KW = dict(capsize=2.5, capthick=0.9, elinewidth=0.9, alpha=0.9)
MARKER = 'D'


def _style_line_axis(ax):
    """Apply the shared log/linear x treatment and light grid to a line axis."""
    if USE_LOG_X:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=12))
    else:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.grid(True, which='major')
    ax.grid(True, which='minor', alpha=0.4)


def _panel_label(ax, text):
    """Bold (a)/(b)/... label above the top-left corner (never clips/overlaps)."""
    ax.set_title(text, loc='left', fontweight='bold', fontsize=12, pad=6)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_timings():
    df = pd.read_csv('timings.csv', na_values=['#DIV/0!'])
    df['Population'] = df['Box Width'] ** 2
    return df


def load_results():
    df = pd.read_csv('results.csv')
    df.loc[df['Dimensions'] == '2D', 'Population'] = df.loc[df['Dimensions'] == '2D', 'Box width'] ** 2
    df.loc[df['Dimensions'] == '3D', 'Population'] = df.loc[df['Dimensions'] == '3D', 'Box width'] ** 3
    return df


def _save(fig, stem):
    if SAVE_PDF:
        fig.savefig(f'./{stem}.pdf')
    if SAVE_PNG:
        fig.savefig(f'./{stem}.png')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 1 — runtime profiling (2x2)
# --------------------------------------------------------------------------- #
def _panel_mechanics_proportion(ax):
    df = load_results()
    cpu2d = df[(df['Simulator'] == 'cpu') & (df['Dimensions'] == '2D')].copy()
    cpu2d = cpu2d.sort_values('Population Size')
    proportion = 100.0 * cpu2d['Mechanics Time'] / cpu2d['Total Time']
    ax.plot(cpu2d['Population Size'], proportion, color=C2D, marker=MARKER)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Population')
    ax.set_ylabel('Time on mechanics (%)')
    _style_line_axis(ax)


def _panel_cpu_proportion(ax):
    df = load_timings()
    df = df[df['Box Width'] >= 50]
    ax.errorbar(df['Population'], df['Proportion CPU Time'] * 100.0,
                yerr=df['CPU Error'] * 100.0, color=OKABE['orange'],
                marker=MARKER, **ERRORBAR_KW)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Population')
    ax.set_ylabel('Time on CPU activities (%)')
    _style_line_axis(ax)


def _panel_gpu_composition(ax):
    df = load_timings()
    df = df[df['Box Width'] >= 200].copy()
    populations = df['Population'].to_numpy()
    translation = df['Wrangle Prop'].to_numpy()
    transfer = df['Transfer Prop'].to_numpy()
    simulation = 1.0 - translation - transfer              # remainder

    x = np.arange(len(populations))
    ax.bar(x, translation, color=STAGE_COLOURS['Device to host data translation'],
           label='Data translation', width=0.8, linewidth=0)
    ax.bar(x, transfer, bottom=translation, color=STAGE_COLOURS['Host to device transfer'],
           label='Data transfer', width=0.8, linewidth=0)
    ax.bar(x, simulation, bottom=translation + transfer, color=STAGE_COLOURS['Simulation'],
           label='Simulation', width=0.8, linewidth=0)

    # Categorical axis: show populations against a common 10**5 factor so the
    # tick labels stay short and horizontal (no overlap).
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p / 1e5:g}' for p in populations])
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'Population ($\times 10^{5}$)')
    ax.set_ylabel('Proportion of GPU runtime')
    ax.grid(True, axis='y')
    ax.grid(False, axis='x')
    handles, labels = ax.get_legend_handles_labels()   # top-to-bottom to match stack
    ax.legend(handles[::-1], labels[::-1], loc='lower left')


def _panel_total_runtime(ax):
    df = load_timings()
    # (colour key, short legend label, series). D2H = Device to Host, H2D = Host to Device.
    bands = [
        ('Simulation',                      'Simulation',      df['Simulation']),
        ('CPU time',                        'CPU',             df['CPU Time']),
        ('Device to host data translation', 'D2H translation', df['DTH Wrangle']),
        ('Device to host transfer',         'D2H transfer',    df['DTH Transfer']),
        ('Host to device transfer',         'H2D transfer',    df['HTD Transfer']),
        ('Host to device data translation', 'H2D translation', df['HTD Wrangle']),
    ]
    ax.stackplot(df['Population'], [b[2] for b in bands],
                 labels=[b[1] for b in bands],
                 colors=[STAGE_COLOURS[b[0]] for b in bands], linewidth=0)
    ax.set_xlim(0, 1_000_000)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Population')
    ax.set_ylabel('Time (ms)')
    for axis in (ax.xaxis, ax.yaxis):                  # scientific notation
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        axis.set_major_formatter(fmt)
    ax.legend(loc='upper left')


def make_profiling_figure():
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    _panel_mechanics_proportion(axes[0, 0]); _panel_label(axes[0, 0], '(A)')
    _panel_cpu_proportion(axes[0, 1]);       _panel_label(axes[0, 1], '(B)')
    _panel_gpu_composition(axes[1, 0]);      _panel_label(axes[1, 0], '(C)')
    _panel_total_runtime(axes[1, 1]);        _panel_label(axes[1, 1], '(D)')
    _save(fig, 'fig-profiling')


# --------------------------------------------------------------------------- #
# Figure 2 — speedup (1x2)
# --------------------------------------------------------------------------- #
def _gpu_rows(df, dim):
    return df[(df['Simulator'] == 'gpu') & (df['Dimensions'] == dim)]


def _panel_mechanics_speedup(ax):
    df = load_results()
    d2, d3 = _gpu_rows(df, '2D'), _gpu_rows(df, '3D')
    e2 = pd.read_csv('mech_errors_2d.csv')['error']
    e3 = pd.read_csv('mech_errors_3d.csv')['error']
    ax.errorbar(d2['Population'], d2['Mechanics Speedup'], yerr=e2, color=C2D,
                marker=MARKER, label='2D', **ERRORBAR_KW)
    ax.errorbar(d3['Population'], d3['Mechanics Speedup'], yerr=e3, color=C3D,
                marker=MARKER, label='3D', **ERRORBAR_KW)
    ax.set_xlabel('Population')
    ax.set_ylabel('Mechanics speedup')
    ax.set_ylim(bottom=0)
    _style_line_axis(ax)
    ax.legend(loc='upper left')


def _panel_total_speedup(ax):
    df = load_results()
    d2, d3 = _gpu_rows(df, '2D'), _gpu_rows(df, '3D')
    e2 = pd.read_csv('total_speedup_errors_2d.csv')
    e3 = pd.read_csv('total_speedup_errors_3d.csv')
    # Actual = solid, maximum theoretical = dashed; 2D blue, 3D vermillion.
    ax.errorbar(d2['Population'], d2['Actual Speedup'], yerr=e2['l1e'], color=C2D,
                marker=MARKER, label='Actual (2D)', **ERRORBAR_KW)
    ax.errorbar(d2['Population'], d2['Maximum Theoretical Speedup'], yerr=e2['l2e'], color=C2D,
                marker=MARKER, linestyle='--', label='Max. theoretical (2D)', **ERRORBAR_KW)
    ax.errorbar(d3['Population'], d3['Actual Speedup'], yerr=e3['l3e'], color=C3D,
                marker=MARKER, label='Actual (3D)', **ERRORBAR_KW)
    ax.errorbar(d3['Population'], d3['Maximum Theoretical Speedup'], yerr=e3['l4e'], color=C3D,
                marker=MARKER, linestyle='--', label='Max. theoretical (3D)', **ERRORBAR_KW)
    ax.set_xlabel('Population')
    ax.set_ylabel('Speedup')
    ax.set_ylim(bottom=0)
    _style_line_axis(ax)
    # Curves plateau high on the right, leaving the lower-right corner clear.
    ax.legend(loc='lower right')


def make_speedup_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    _panel_total_speedup(axes[0]);     _panel_label(axes[0], '(A)')
    _panel_mechanics_speedup(axes[1]); _panel_label(axes[1], '(B)')
    _save(fig, 'fig-speedup')


# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    make_profiling_figure()
    make_speedup_figure()
