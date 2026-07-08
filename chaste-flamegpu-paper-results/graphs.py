import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import random

sns.set_theme()#palette='viridis')
mpl.rcParams['figure.dpi'] = 300
marker_style = dict(marker='D', markersize=4, markeredgewidth=0, ci="sd")


def plot_mechanics_proportion():
    # Mechanics proportion
    mechanics_prop_raw_data = pd.read_csv('mechanics-prop.csv')
    mech_errors = mechanics_prop_raw_data['Error']
    mechanics_prop_plot = sns.lineplot(data=mechanics_prop_raw_data, x='Box Size', y='Time Spent on Mechanics (%)', errorbar="se", **marker_style)
    mechanics_prop_plot.set_ylim(bottom=0.0, top=100.0)
    mechanics_prop_plot.set_xlabel("Box Width")
    mechanics_prop_plot.set_ylabel("Time spent on mechanics (%)")
    mechanics_prop_plot.errorbar(mechanics_prop_raw_data['Box Size'], mechanics_prop_raw_data['Time Spent on Mechanics (%)'], yerr=mech_errors, color='blue', alpha=0.8)
    plt.savefig('./mech-time.png')

# Timings

def plot_cpu_time():
    timings_raw_data = pd.read_csv('timings.csv', na_values=['#DIV/0!'])
    timings_raw_data['Population'] = timings_raw_data['Box Width'] * timings_raw_data['Box Width']

    cput_data = timings_raw_data[timings_raw_data['Box Width'] >= 50]
    cput_errors = cput_data['CPU Error']
    cpu_time_plot = sns.lineplot(data=cput_data, x='Population', y='Proportion CPU Time', errorbar="se", **marker_style)
    cpu_time_plot.set_ylim(bottom=0.0, top=1.0)
    cpu_time_plot.set_xlabel("Box Width")
    cpu_time_plot.set_ylabel("Time spent on CPU activities (%)")

    cpu_time_plot.errorbar(cput_data['Population'], cput_data['Proportion CPU Time'], yerr=cput_errors, color='orange', alpha=0.8)
    plt.ticklabel_format(style='plain', axis='x')
    plt.savefig('./cpu-time.png')

def plot_gpu_time():
    timings_raw_data = pd.read_csv('timings.csv', na_values=['#DIV/0!'])
    timings_raw_data['Population'] = timings_raw_data['Box Width'] * timings_raw_data['Box Width']
    timings_raw_data['Sim Prop'] = 1.0
    timings_raw_data['Transfer Prop'] = timings_raw_data['Transfer Prop'] + timings_raw_data['Wrangle Prop']

    cmap = plt.cm.get_cmap('tab10')
    top_bar = sns.barplot(data=timings_raw_data[timings_raw_data['Box Width'] >= 200], x='Population', y='Sim Prop', color=cmap(0), linewidth=0)
    middle_bar = sns.barplot(data=timings_raw_data[timings_raw_data['Box Width'] >= 200], x='Population', y='Transfer Prop', color=cmap(1), linewidth=0)
    bottom_bar = sns.barplot(data=timings_raw_data[timings_raw_data['Box Width'] >= 200], x='Population', y='Wrangle Prop', color=cmap(2), linewidth=0)

    top_legend = mpatches.Patch(color=cmap(0), label='Simulation')
    mid_legend = mpatches.Patch(color=cmap(1), label='Data transfer')
    bot_legend = mpatches.Patch(color=cmap(2), label='Data translation')
    plt.legend(handles=[top_legend, mid_legend, bot_legend])
    plt.ylabel('Proportion of GPU runtime')
    plt.savefig('./gpu-time.png')
    plt.close()

def plot_total_runtime():
    ## Total Runtime
    fig = plt.figure()

    timings_raw_data = pd.read_csv('timings.csv', na_values=['#DIV/0!'])
    timings_raw_data['Population'] = timings_raw_data['Box Width'] * timings_raw_data['Box Width']
    timings_raw_data['Sim Prop'] = 1.0
    timings_raw_data['Transfer Prop'] = timings_raw_data['Transfer Prop'] + timings_raw_data['Wrangle Prop']

    total_runtime_y_data = [
        timings_raw_data['Simulation'],
        timings_raw_data['CPU Time'],
        timings_raw_data['DTH Wrangle'],
        timings_raw_data['DTH Transfer'],
        timings_raw_data['HTD Transfer'],
        timings_raw_data['HTD Wrangle'],
    ]

    plt.stackplot(timings_raw_data['Population'], total_runtime_y_data, labels=['Device to host data translation', 'Device to host transfer', 'Host to device data translation', 'Host to device transfer', 'Simulation', 'CPU time'], linewidth=0)
    plt.legend(loc='upper left')
    plt.ticklabel_format(style='plain', axis='x')
    plt.xlabel('Population')
    plt.ylabel('Time (ms)')
    plt.xlim(left=0, right=1000000)
    #fig.axes[0].set_xscale('log')
    fig.set_tight_layout(True)
    plt.savefig('./overall-time.png')
    plt.close()

# Performance
def plot_mechanics_speedup():
    raw_data = pd.read_csv('results.csv')
    raw_data.loc[raw_data['Dimensions'] == '2D', 'Population'] = pow(raw_data[raw_data['Dimensions'] == '2D']['Box width'], 2)
    raw_data.loc[raw_data['Dimensions'] == '3D', 'Population'] = pow(raw_data[raw_data['Dimensions'] == '3D']['Box width'], 3)

    ## Mechanics Speedup
    mechs_data_2D = raw_data[raw_data['Dimensions'] == '2D'][raw_data['Simulator'] == 'gpu']
    mechs_errors_2D = pd.read_csv('mech_errors_2d.csv')['error']
    mech_speedup_plot = sns.lineplot(data=raw_data[raw_data['Dimensions'] == '2D'], x='Population', y='Mechanics Speedup', label='2D', errorbar="se", **marker_style)
    mech_speedup_plot.errorbar(mechs_data_2D['Population'], mechs_data_2D['Mechanics Speedup'], yerr=mechs_errors_2D, color='b', alpha=0.8)
    mechs_data_3D = raw_data[raw_data['Dimensions'] == '3D'][raw_data['Simulator'] == 'gpu']
    mechs_errors_3D = pd.read_csv('mech_errors_3d.csv')['error']
    mech_speedup_plot = sns.lineplot(data=raw_data[raw_data['Dimensions'] == '3D'], x='Population', y='Mechanics Speedup', label='3D', errorbar="se", **marker_style)
    mech_speedup_plot.errorbar(mechs_data_3D['Population'], mechs_data_3D['Mechanics Speedup'], yerr=mechs_errors_3D, color='orange', alpha=0.8)
    plt.ticklabel_format(style='plain', axis='x')
    plt.xlabel('Population')
    plt.ylabel('Speedup')
    plt.savefig('./mech-speedup.png')
    plt.close()

def plot_total_speedup():
    raw_data = pd.read_csv('results.csv')
    raw_data.loc[raw_data['Dimensions'] == '2D', 'Population'] = pow(raw_data[raw_data['Dimensions'] == '2D']['Box width'], 2)
    raw_data.loc[raw_data['Dimensions'] == '3D', 'Population'] = pow(raw_data[raw_data['Dimensions'] == '3D']['Box width'], 3)

    ## Total speedup
    l1d = raw_data.query('Simulator == "gpu" & Dimensions == "2D"')
    l2d = raw_data.query('Simulator == "gpu" & Dimensions == "2D"')
    l3d = raw_data.query('Simulator == "gpu" & Dimensions == "3D"')
    l4d = raw_data.query('Simulator == "gpu" & Dimensions == "3D"')

    raw_errors = pd.read_csv('total_speedup_errors.csv')
    mechs_errors_2D = pd.read_csv('mech_errors_2d.csv')['error']
    mech_speedup_plot = sns.lineplot(data=raw_data[raw_data['Dimensions'] == '2D'], x='Population', y='Mechanics Speedup', label='2D', errorbar="se", **marker_style)
    mech_speedup_plot.errorbar(mechs_data_2D['Population'], mechs_data_2D['Mechanics Speedup'], yerr=mechs_errors_2D, color='b', alpha=0.8)
    mechs_data_3D = raw_data[raw_data['Dimensions'] == '3D'][raw_data['Simulator'] == 'gpu']
    mechs_errors_3D = pd.read_csv('mech_errors_3d.csv')['error']
    mech_speedup_plot = sns.lineplot(data=raw_data[raw_data['Dimensions'] == '3D'], x='Population', y='Mechanics Speedup', label='3D', errorbar="se", **marker_style)
    mech_speedup_plot.errorbar(mechs_data_3D['Population'], mechs_data_3D['Mechanics Speedup'], yerr=mechs_errors_3D, color='orange', alpha=0.8)
    plt.ticklabel_format(style='plain', axis='x')
    plt.xlabel('Population')
    plt.ylabel('Speedup')
    plt.savefig('./mech-speedup.png')
    plt.close()

def plot_total_speedup():
    raw_data = pd.read_csv('results.csv')
    raw_data.loc[raw_data['Dimensions'] == '2D', 'Population'] = pow(raw_data[raw_data['Dimensions'] == '2D']['Box width'], 2)
    raw_data.loc[raw_data['Dimensions'] == '3D', 'Population'] = pow(raw_data[raw_data['Dimensions'] == '3D']['Box width'], 3)

    ## Total speedup
    l1d = raw_data.query('Simulator == "gpu" & Dimensions == "2D"')
    l2d = raw_data.query('Simulator == "gpu" & Dimensions == "2D"')
    l3d = raw_data.query('Simulator == "gpu" & Dimensions == "3D"')
    l4d = raw_data.query('Simulator == "gpu" & Dimensions == "3D"')

    raw_errors_2d = pd.read_csv('total_speedup_errors_2d.csv')
    raw_errors_3d = pd.read_csv('total_speedup_errors_3d.csv')

    l1e = raw_errors_2d["l1e"]
    l2e = raw_errors_2d["l2e"]
    l3e = raw_errors_3d["l3e"]
    l4e = raw_errors_3d["l4e"]

    overall_speedup_plot = sns.lineplot(data=raw_data.query('Simulator == "gpu" & Dimensions == "2D"'), errorbar="se", x='Population', y='Actual Speedup', label='Actual speedup (2D)', **marker_style)
    overall_speedup_plot = sns.lineplot(data=raw_data.query('Simulator == "gpu" & Dimensions == "2D"'), errorbar="se", x='Population', y='Maximum Theoretical Speedup', linestyle='--', label='Maximum theoretical speedup (2D)', **marker_style)
    overall_speedup_plot = sns.lineplot(data=raw_data.query('Simulator == "gpu" & Dimensions == "3D"'), errorbar="se", x='Population', y='Actual Speedup', label='Actual speedup (3D)', **marker_style)
    overall_speedup_plot = sns.lineplot(data=raw_data.query('Simulator == "gpu" & Dimensions == "3D"'), errorbar="se", x='Population', y='Maximum Theoretical Speedup', linestyle='--', label='Maximum theoretical speedup (3D)', **marker_style)

    overall_speedup_plot.errorbar(l1d['Population'], l1d['Actual Speedup'], yerr=l1e, color='b', alpha=0.8)
    overall_speedup_plot.errorbar(l2d['Population'], l2d['Maximum Theoretical Speedup'], yerr=l2e, color='orange', alpha=0.8)
    overall_speedup_plot.errorbar(l3d['Population'], l3d['Actual Speedup'], yerr=l3e, color='green', alpha=0.8)
    overall_speedup_plot.errorbar(l4d['Population'], l4d['Maximum Theoretical Speedup'], yerr=l4e, color='red', alpha=0.8)

    plt.ticklabel_format(style='plain', axis='x')
    plt.xlabel('Population')
    plt.ylabel('Speedup')
    plt.savefig('./overall-speedup.png')


# Produce the plots
plot_mechanics_proportion()
plot_cpu_time()
plot_gpu_time()
plot_total_runtime()
plot_mechanics_speedup()
plot_total_speedup()