import matplotlib.pyplot as plt
import numpy as np

cases = [
    'Aero1',
    'Aero2',
    'Aero3',
    'Aero4',
    'Aero5'
]

cbs = [
    2334.54,
    1995.88,
    2523.55,
    2486.34,
    2701.48
]

sp = [
    273.46,
    211.46,
    306.12,
    310.47,
    297.51
]

x = np.arange(len(cases))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width/2,
    cbs,
    width,
    label='CBS WCD'
)

plt.bar(
    x + width/2,
    sp,
    width,
    label='SP WCD'
)

plt.xticks(x, cases)

plt.ylabel('Delay (µs)')
plt.xlabel('Benchmark Case')

plt.title(
    'Analytical Worst-Case Delays'
)

plt.legend()

plt.tight_layout()

plt.savefig(
    'figures/analytical_delays.png',
    dpi=300
)

plt.close()


sim_max = [
    679.26,
    396.57,
    835.03,
    846.20,
    687.23
]

sim_avg = [
    219.09,
    98.82,
    199.56,
    152.62,
    200.20
]

plt.figure(figsize=(8, 5))

plt.bar(
    x - width/2,
    sim_max,
    width,
    label='Simulated Max'
)

plt.bar(
    x + width/2,
    sim_avg,
    width,
    label='Simulated Avg'
)

plt.xticks(x, cases)

plt.ylabel('Delay (µs)')
plt.xlabel('Benchmark Case')

plt.title(
    'Simulation Results'
)

plt.legend()

plt.tight_layout()

plt.savefig(
    'figures/simulation_results.png',
    dpi=300
)

plt.close()


plt.figure(figsize=(8, 5))

plt.bar(
    x - width/2,
    cbs,
    width,
    label='CBS Analytical'
)

plt.bar(
    x + width/2,
    sim_max,
    width,
    label='Simulated Max'
)

plt.xticks(x, cases)

plt.ylabel('Delay (µs)')
plt.xlabel('Benchmark Case')

plt.title(
    'Analytical vs Simulated Delays'
)

plt.legend()

plt.tight_layout()

plt.savefig(
    'figures/analytical_vs_simulated.png',
    dpi=300
)

plt.close()

plt.figure(figsize=(8, 5))

plt.bar(
    x - width/2,
    cbs,
    width,
    label='CBS'
)

plt.bar(
    x + width/2,
    sp,
    width,
    label='SP'
)

plt.xticks(x, cases)

plt.ylabel('Delay (µs)')
plt.xlabel('Benchmark Case')

plt.title(
    'CBS vs SP Comparison'
)

plt.legend()

plt.tight_layout()

plt.savefig(
    'figures/cbs_vs_sp.png',
    dpi=300
)

plt.close()