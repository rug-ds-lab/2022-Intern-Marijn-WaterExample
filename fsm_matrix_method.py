import pickle

import pandas as pd
import wntr
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re


def run(network_path, data_path):
    pressures = read_pressure_as_matrix(data_path)

    no_leak_signature = run_scenario(network_path)
    no_leak_signature.index = no_leak_signature.index.astype(int)
    no_leak_signature = no_leak_signature.sort_index()
    s_matrix = generate_sensitivity_matrix(network_path)

    s_matrix = s_matrix.drop(columns=[1])
    no_leak_signature = no_leak_signature.drop(labels=[1])
    pressures = pressures.drop(columns=[1])

    correlation_signal = []

    for t, p in enumerate(pressures.values):
        print(f'time: {t}')
        correlation_signal.append(
            leak_detection_fsm_matrix(p, s_matrix, no_leak_signature)
        )

    with open('model_data/correlation_signal.pkl', 'wb') as f:
        pickle.dump(correlation_signal, f)

    w = 40
    cumulative_correlation = []

    for i in range(w, len(correlation_signal)):
        past_values = [1 if j < 0.5 else 0 for j in [correlation_signal[k] for k in range(i - w, i)]]
        if sum(past_values) == 0:
            # This can be made more optimal since if a single '1' is appended we can already go to the next iteration
            cumulative_correlation.append(1)
        else:
            cumulative_correlation.append(0)

    plt.title(data_path + ': Filtered signal')
    plt.plot(cumulative_correlation)
    plt.show()


def read_pressure_as_matrix(folder_path):
    pressures = pd.DataFrame()
    for link_path in glob.glob(folder_path + '/Node_*.csv'):
        node_pressure = pd.read_csv(link_path)['Value']
        node_n = int(re.sub('\D', '', os.path.basename(link_path)))
        pressures[node_n] = node_pressure
    return pressures.reindex(sorted(pressures.columns), axis=1)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def leak_detection_fsm_matrix(current_pressure, sensitivity_matrix, no_leak_signature):
    fsm_matrix = generate_fsm_matrix(sensitivity_matrix, no_leak_signature.T, current_pressure)
    correlations = [c[0][1] + c[1][0] for c in fsm_matrix]
    leak_prediction = np.argmax(correlations)
    if correlations[leak_prediction] > 1.9:
        print(f'Highest leak chance at: {leak_prediction + 1}')
        print(f'Correlation value: {correlations[leak_prediction]}')
        print(correlations)

    return correlations[leak_prediction]


def compute_correlation(v1, v2):
    correlation_matrix = np.corrcoef(v1, v2)
    return correlation_matrix


def generate_fsm_matrix(sensitivity_matrix, no_leak_signature, current_pressures):
    residual_vector = current_pressures - no_leak_signature
    fsm_matrix = \
        [compute_correlation(leak_signature.T, residual_vector) for leak_signature in sensitivity_matrix.values]
    return fsm_matrix


def generate_sensitivity_matrix(network_path):
    wn = wntr.network.WaterNetworkModel(network_path)
    fault_signature_matrix = \
        pd.DataFrame([run_scenario(
            input_network_path=network_path,
            leak_scenario=True,
            node_name=node_name
        ) for node_name in wn.node_name_list if node_name != '1'])
    no_leak_signature = run_scenario(network_path)
    sensitivity_matrix = fault_signature_matrix - no_leak_signature.values.squeeze()
    return sensitivity_matrix


def run_scenario(input_network_path, leak_scenario=False, node_name=None):
    print(f'Simulating leak at node: {node_name}')
    wn = wntr.network.WaterNetworkModel(input_network_path)
    sim_step_seconds = 5
    duration_hours = 0
    wn.options.time.duration = duration_hours * 3600
    wn.options.time.hydraulic_timestep = sim_step_seconds
    wn.options.time.quality_timestep = 0
    wn.options.time.report_timestep = sim_step_seconds
    wn.options.time.pattern_timestep = sim_step_seconds

    if leak_scenario is True:
        node = wn.get_node(node_name)
        node.add_leak(wn, area=0.5, start_time=0,
                      end_time=wn.options.time.duration - 3 * wn.options.time.hydraulic_timestep)

    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    pressure = results.node['pressure']
    leak_signature = pressure.loc[0, :]
    leak_signature.index = leak_signature.index.astype(int)

    return leak_signature.sort_index()


if __name__ == '__main__':
    run('network_input_files/Hanoi_CMH.inp', 'dataset/Scenario-13/Pressures')
