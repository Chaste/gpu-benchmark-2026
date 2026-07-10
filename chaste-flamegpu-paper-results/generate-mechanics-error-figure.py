"""Generate the mechanical-substep convergence figure (manuscript Figure 3).

Two particles connected by a spring (the node-based mechanics model used in
Chaste) are integrated forward with an explicit Euler scheme, using a range of
substep counts per biological timestep. A very finely resolved solve
(GROUND_TRUTH_ITERATIONS substeps) is taken as the ground truth, and the
relative error in the position of one particle is computed against it over time.

The figure is written as both a vector PDF (for submission) and a 300-dpi PNG
(for preview). Run as a script with no arguments::

    python generate_mechanics_error_figure.py

Styling is kept consistent with the profiling and speedup figures (sans-serif
font, light grid, no top/right spines, scientific-notation axis, vector output).
The one deliberate departure is colour: the substep counts form an *ordered*
family, so they are drawn with a perceptually uniform sequential colour map
(darker = fewer substeps, larger error) rather than the categorical Okabe-Ito
palette used for the 2D/3D series elsewhere.

Note on the y-axis: the plotted quantity is a fraction (ground truth - trial) /
trial, i.e. a relative error, not a percentage. The axis is labelled accordingly;
if a true percentage is wanted, scale the errors by 100 and update the caption.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
SAVE_PDF = True
SAVE_PNG = True

GROUND_TRUTH_ITERATIONS = 20000
TRIAL_SUBSTEPS = [1, 2, 5, 10, 20, 60, 120, 300]

# Two-particle spring model parameters
INITIAL_POSITIONS = (-0.7, 0.7)     # (p1, p2)
REST_LENGTHS = (0.5, 0.5)           # (r1, r2)
SPRING_CONSTANT = 15.0
END_TIME = 0.5                      # hours
DT = 1.0 / 120.0                   # biological timestep

# --------------------------------------------------------------------------- #
# Global style — matches make_publication_figures.py
# --------------------------------------------------------------------------- #
mpl.rcParams.update({
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'Liberation Sans',
                          'Nimbus Sans', 'DejaVu Sans'],
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
    'lines.linewidth':   1.4,
    'axes.linewidth':    0.8,
    'axes.grid':         True,
    'grid.color':        '0.85',
    'grid.linewidth':    0.6,
    'axes.axisbelow':    True,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def timepoints():
    """Return the list of biological-timestep times, in hours."""
    times, t = [], 0.0
    while t < END_TIME:
        times.append(t)
        t += DT
    return times


def simulate(substeps):
    """Integrate the two-particle spring model and return p1 at each timestep.

    Each biological timestep of length DT is advanced with `substeps` explicit
    Euler substeps.
    """
    p1, p2 = INITIAL_POSITIONS
    r1, r2 = REST_LENGTHS
    k = SPRING_CONSTANT
    rest_length = r1 + r2

    positions, t = [], 0.0
    while t < END_TIME:
        for _ in range(substeps):
            overlap = (p2 - p1) - rest_length
            if overlap < 0:
                f = k * rest_length * math.log(1.0 + (overlap / rest_length))
            else:
                f = k * overlap * math.exp(-5.0 * (overlap / rest_length))
            # Euler integration
            p1 += f * (DT / substeps)
            p2 -= f * (DT / substeps)
        t += DT
        positions.append(p1)
    return positions


def relative_errors():
    """Return {substeps: [relative error at each timestep]} against ground truth."""
    ground_truth = simulate(GROUND_TRUTH_ITERATIONS)
    errors = {}
    for substeps in TRIAL_SUBSTEPS:
        trial = simulate(substeps)
        errors[substeps] = [(gt - val) / val for gt, val in zip(ground_truth, trial)]
    return errors


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure():
    times = timepoints()
    errors = relative_errors()

    # Ordered family -> perceptually uniform sequential colours (dark = fewer
    # substeps). Truncated to avoid the palest, low-contrast end of the map.
    cmap = plt.get_cmap('viridis')
    colours = [cmap(x) for x in np.linspace(0.0, 0.9, len(TRIAL_SUBSTEPS))]

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    for substeps, colour in zip(TRIAL_SUBSTEPS, colours):
        ax.plot(times, errors[substeps], color=colour, label=str(substeps))

    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Relative error in position')
    ax.set_xlim(left=0.0)

    # Scientific notation for the small error values.
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)

    ax.legend(title='Mechanical substeps per timestep', loc='lower right',
              ncol=2, labelspacing=0.3, handlelength=1.6, borderpad=0.6)

    if SAVE_PDF:
        fig.savefig('./fig-mechanics-error.pdf')
    if SAVE_PNG:
        fig.savefig('./fig-mechanics-error.png')
    plt.close(fig)


if __name__ == '__main__':
    make_figure()
