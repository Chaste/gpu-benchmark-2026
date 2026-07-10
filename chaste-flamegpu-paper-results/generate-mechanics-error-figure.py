"""
Generate a figure showing the error in position of two particles connected by a spring,
for different numbers of substeps per timestep. Based on the mechanics model used in
Chaste for node-based simulations. An explicit solve with a very high number of
iterations (i.e. a very fine timestep) is used as the ground truth, and the error
in position is computed against this.
"""

import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 120
sns.set_theme()
marker_style = dict(marker='D', markersize=4, markeredgewidth=0)

GROUND_TRUTH_ITERATIONS = 20000
trial_substeps = [1, 2, 5, 10, 20, 60, 120, 300, GROUND_TRUTH_ITERATIONS]
results = {}

for trial_substep in trial_substeps:

    # Initial positions
    p1 = -0.7
    p2 = 0.7

    # Rest lengths
    r1 = 0.5
    r2 = 0.5

    # Spring constant
    k = 15.0

    # Time parameters
    t = 0
    end_time = 0.5
    dt = 1.0 / 120.0

    # Produce the time points for plotting
    times = []
    while t < end_time:
        times.append(t)
        t += dt
    t  = 0

    substeps = trial_substep
    results[trial_substep] = []

    while t < end_time:

        # For each substep, calculate the force and update the positions of the two particles
        for substep in range(0, substeps):
            dist = p2 - p1
            rest_length = r1 + r2
            overlap = dist - rest_length
            f = 0

            if overlap < 0:
                f = k * rest_length * math.log(1.0 + (overlap / rest_length))
            else:
                f = k * overlap * math.exp(-5.0  * (overlap / rest_length))

            # Euler integration
            p1 += f * (dt / substeps)
            p2 -= f * (dt / substeps)

        t += dt
        results[trial_substep].append(p1)

# Initialise arrays to store the errors
errors = {}

for trial_substep in trial_substeps:
    errors[trial_substep] = []

# Compute the percentage error in position for each trial substep compared to the ground truth
i = 0
for v in results[GROUND_TRUTH_ITERATIONS]:
    for trial_substep in trial_substeps:
        if trial_substep != GROUND_TRUTH_ITERATIONS:
            actual_val = results[trial_substep][i]
            errors[trial_substep].append((v - actual_val) / actual_val)
    i += 1


# Plot the errors
sns.set_theme()

plt.figure()
for trial_substep in trial_substeps:
    if trial_substep != GROUND_TRUTH_ITERATIONS:
        plt.plot(times, errors[trial_substep], label=str(trial_substep), **marker_style)
plt.xlabel("Time (hours)")
plt.ylabel("Percentage error in position")
plt.legend(title="Mechanical substeps per timestep", loc="lower right", fontsize=8,
           title_fontsize=8, markerscale=0.7, labelspacing=0.3, handlelength=1.5, borderpad=0.5)
plt.savefig("multiple-steps.png", bbox_inches="tight")