import wntr
import pandas as pd
from configparser import ConfigParser
import pickle
import os


def generate_signature(input_network_path, leak_scenario=False, node_name=None):
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


def generate_sensitivity_matrix(network_path, no_leak_signature):
    # This function relies heavily on the assumption that node 1 is the reservoir
    wn = wntr.network.WaterNetworkModel(network_path)
    fault_signature_matrix = \
        pd.DataFrame([generate_signature(
            input_network_path=network_path,
            leak_scenario=True,
            node_name=node_name
        ) for node_name in wn.node_name_list if node_name != '1'])

    sensitivity_matrix = fault_signature_matrix - no_leak_signature.values.squeeze()
    sensitivity_matrix = sensitivity_matrix.drop(columns=[1])
    return sensitivity_matrix


def run():
    config = ConfigParser()
    config.read('config.ini')
    input_network_path = config['DEFAULT']['input_network_path']

    no_leak_signature = generate_signature(input_network_path)
    sensitivity_matrix = generate_sensitivity_matrix(input_network_path, no_leak_signature)

    with open('model/sensitivity_matrix.pkl', 'wb') as f:
        pickle.dump(sensitivity_matrix, f)

    with open('model/no_leak_signature.pkl', 'wb') as f:
        pickle.dump(no_leak_signature.iloc[1:], f)


if __name__ == '__main__':
    run()
