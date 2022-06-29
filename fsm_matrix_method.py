import pandas as pd
import wntr
import matplotlib.pyplot as plt
import numpy as np


def run(network_path):
    wn = wntr.network.WaterNetworkModel(network_path)
    node = wn.get_node('5')
    sim_step_seconds = 5
    duration_hours = 0
    wn.options.time.duration = duration_hours * 3600
    wn.options.time.hydraulic_timestep = sim_step_seconds
    wn.options.time.quality_timestep = 0
    wn.options.time.report_timestep = sim_step_seconds
    wn.options.time.pattern_timestep = sim_step_seconds
    node.add_leak(wn, area=0.05, start_time=0,
                  end_time=wn.options.time.duration - 3 * wn.options.time.hydraulic_timestep)
    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()
    initial_pressure = results.node['pressure'].loc[0, :]
    print(f'Initial pressure: {initial_pressure}')

    no_leak_signature = run_scenario(network_path)
    s_matrix = generate_sensitivity_matrix(network_path)

    generate_fsm_matrix(s_matrix, no_leak_signature.T, initial_pressure)


def compute_correlation(v1, v2):
    print(f'Leak signature: {v1}')
    print(f'Current pressure: {v2}')
    correlation = (np.cov(v1, v2) / (np.sqrt(np.cov(v1, v1)*np.cov(v2, v2).T)))
    print(f'Correlation: {correlation}')
    return correlation


def generate_fsm_matrix(sensitivity_matrix, no_leak_signature, current_pressures):
    residual_vector = current_pressures - no_leak_signature
    print(f'Residual: {residual_vector}')
    print(f'Current pressure: {current_pressures}')
    fsm_matrix = \
        [compute_correlation(leak_signature.T, residual_vector) for leak_signature in sensitivity_matrix.values]
    print(fsm_matrix)


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
    wntr.graphics.plot_network(wn, title=wn.name)
    plt.show()

    sim_step_seconds = 5
    duration_hours = 0
    wn.options.time.duration = duration_hours * 3600
    wn.options.time.hydraulic_timestep = sim_step_seconds
    wn.options.time.quality_timestep = 0
    wn.options.time.report_timestep = sim_step_seconds
    wn.options.time.pattern_timestep = sim_step_seconds

    if leak_scenario is True:
        node = wn.get_node(node_name)
        node.add_leak(wn, area=0.05, start_time=0,
                      end_time=wn.options.time.duration - 3 * wn.options.time.hydraulic_timestep)

    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    initial_pressure = results.node['pressure'].loc[0, :]
    wntr.graphics.plot_network(wn, node_attribute=initial_pressure, node_size=30,
                               title=f'Pressure at t={duration_hours}')
    plt.show()
    pressure = results.node['pressure']
    leak_signature = pressure.loc[0, :]

    return leak_signature


if __name__ == '__main__':
    run('network_data/Hanoi_CMH.inp')
