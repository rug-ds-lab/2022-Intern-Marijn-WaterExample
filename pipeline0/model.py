import wntr
import pandas as pd
from configparser import ConfigParser
import pickle
import numpy as np


def generate_binary_leak_prediction(fsm_matrix, threshold):
    """
    Checks if correlation value exceeds a threshold

    :param fsm_matrix: fault sensitivity matrix for a given network
    :param threshold: the correlation threshold value
    :return: boolean value corresponding to whether the threshold was exceeded or not
    """
    corr_sig = [c[0][1] for c in fsm_matrix]
    for node_pressure_correlation in corr_sig:
        if node_pressure_correlation > threshold:
            return True
    return False


def compute_correlation(v1, v2):
    """
    Computes the Pearson correlation coefficients between two vectors

    :param v1:
    :param v2:
    :return: matrix containing the correlation coefficients in the diagonals
    """
    correlation_matrix = np.corrcoef(v1, v2)
    return correlation_matrix


def generate_fsm_matrix(sensitivity_matrix, no_leak_signature, current_pressures):
    """
    Generates the fault sensitivity matrix for pressure values at a given time

    :param sensitivity_matrix: the sensitivity matrix for a given scenario
    :param no_leak_signature: the pressure signature when there is no leak in the network
    :param current_pressures: the current pressure signature in the network
    :return: fault sensitivity matrix
    """
    residual_vector = current_pressures - no_leak_signature
    fsm_matrix = \
        [compute_correlation(leak_signature.T, residual_vector) for leak_signature in sensitivity_matrix.values]
    return fsm_matrix


def generate_signature(input_network_path, leak_scenario=False, node_name=None):
    """
    Runs a very brief water distribution network simulation using wntr with the aim of generating pressure signatures
    for different scenarios

    :param input_network_path:
    :param leak_scenario: whether there should be a leak in the scenario
    :param node_name: which node should contain the leak
    :return: leak signature in the form of a list of pressures
    """
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
    """
    Constructs the sensitivity matrix by running leak simulations for each node in the network

    :param network_path:
    :param no_leak_signature:
    :return: the sensitivity matrix for a given network
    """

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


def train_model():
    """
    Trains the model by generating a sensitivity matrix and a signature for a scenario where there is no leak.

    :return:
    """

    config = ConfigParser()
    config.read('config.ini')
    wdn_input_file_name = config['global']['wdn_input_file_name']
    input_network_path = f'wdn_input_files/{wdn_input_file_name}'

    no_leak_signature = generate_signature(input_network_path)
    sensitivity_matrix = generate_sensitivity_matrix(input_network_path, no_leak_signature)

    with open('model/sensitivity_matrix.pkl', 'wb') as f:
        pickle.dump(sensitivity_matrix, f)

    with open('model/no_leak_signature.pkl', 'wb') as f:
        pickle.dump(no_leak_signature.iloc[1:], f)


def load_model():
    """
    Loads a pre-generated sensitivity matrix and no leak signature from the disk

    :return:
    """

    with open('model/sensitivity_matrix.pkl', 'rb') as f:
        sensitivity_matrix = pickle.load(f)

    with open('model/no_leak_signature.pkl', 'rb') as f:
        no_leak_signature = pickle.load(f)
    return sensitivity_matrix, no_leak_signature
