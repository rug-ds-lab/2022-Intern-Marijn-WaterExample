import json
import os.path
import wntr
from configparser import ConfigParser
import pandas as pd
import numpy as np
import re
import glob

DATASET_DIR = 'dataset'


def read_leakdb_scenario_as_dataframe(folder_path, network_property):
    """
    Reads a scenario using the LeakDB directory structure

    :param folder_path:
    :param network_property: either flow or pressure
    :return: dataframe containing pressure or flow data
    """

    network_data = pd.DataFrame()

    if network_property == 'Pressures':
        file_name = 'Node_*.csv'
    elif network_property == 'Flows':
        file_name = 'Link_*.csv'
    else:
        raise Exception('Incorrect property')

    data_path = os.path.join(folder_path, network_property, file_name)


    for d in glob.glob(data_path):
        node_pressure = pd.read_csv(d)['Value']
        node_n = int(re.sub('\D', '', os.path.basename(d)))
        network_data[node_n] = node_pressure
    return network_data.reindex(sorted(network_data.columns), axis=1)


def train_val_test_split(data, train_start, train_end, val_start, val_end, test_start, test_end):
    """
    Splits data into a train, validation and test set

    :param data:
    :param train_start:
    :param train_end:
    :param val_start:
    :param val_end:
    :param test_start:
    :param test_end:
    :return: train: dataframe containing train data, val: dataframe containing val data, test: dataframe containing
    test data
    """

    train = data[train_start:train_end]
    val = data[val_start:val_end]
    test = data[test_start:test_end]
    return train, val, test


def train_val_test_to_csv(folder_path, train, val, test):
    """
    Saves train, validation and test data as CSV files

    :param folder_path:
    :param train:
    :param val:
    :param test:
    :return:
    """
    train.to_csv(os.path.join(folder_path, 'train.csv'))
    val.to_csv(os.path.join(folder_path, 'val.csv'))
    test.to_csv(os.path.join(folder_path, 'test.csv'))


def save_scenario(folder_path, train_pressure, val_pressure, test_pressure, train_flow, val_flow, test_flow):
    """
    Saves all files corresponding to a simulated scenario

    :param folder_path:
    :param train_pressure:
    :param val_pressure:
    :param test_pressure:
    :param train_flow:
    :param val_flow:
    :param test_flow:
    :return:
    """

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    pressure_path = os.path.join(folder_path, 'pressure')
    if not os.path.exists(pressure_path):
        os.mkdir(pressure_path)

    flow_path = os.path.join(folder_path, 'flow')
    if not os.path.exists(flow_path):
        os.mkdir(flow_path)

    train_val_test_to_csv(pressure_path, train_pressure, val_pressure, test_pressure)
    train_val_test_to_csv(flow_path, train_flow, val_flow, test_flow)


def sort_columns(data):
    """
    Rearranges a Dataframe based on column names. Assumed here is that the column names are integers.

    :param data:
    :return:
    """

    data.columns = data.columns.astype('int64')
    data = data.reindex(sorted(data.columns), axis=1)
    return data


def generate_random_leak_time(simulation_start_time, test_start, test_end, hydraulic_timestep):
    """
    Generates a leak at a random time with a random duration within the test set bounds

    :param simulation_start_time:
    :param test_start:
    :param test_end:
    :param hydraulic_timestep:
    :return: leak start and end indices in wntr format, timestamp format and simple index format
    """

    # Generate timestamps for entire scenario
    timestamps = pd.date_range(start=simulation_start_time, end=test_end, freq=f'{hydraulic_timestep}s')
    timestamps_df = timestamps.to_frame()

    # Add corresponding simple indices
    timestamps_df['ix'] = range(0, len(timestamps_df))

    # Take timestamps corresponding to test set
    possible_leak_times = timestamps_df[test_start:test_end]

    # Sample random timestamp for start of leak
    leak_start = possible_leak_times.sample()

    # Get simple index of random of the leak start time
    leak_start_ix = leak_start['ix'].iloc[0]

    # Store datetime format of leak start time
    leak_start_datetime = leak_start.index

    # Generate random end time of leak between leak start and scenario end time
    leak_end_ix = np.random.randint(low=leak_start_ix, high=len(timestamps_df))

    # Convert simple indices to wntr simulation indices by multiplying them by the simulation's hydraulic timestep
    leak_start_time = leak_start_ix * hydraulic_timestep
    leak_end_time = leak_end_ix * hydraulic_timestep

    return leak_start_time, leak_end_time, leak_start_datetime, leak_start_ix, leak_end_ix


def generate_scenario(is_leak_scenario=False, leak_node=None, scenario_name='scenario-2'):
    """
    Generates flow and pressure data for a water distribution network simulation using the wntr library

    :param is_leak_scenario: whether the scenario should contain a leak or not
    :param leak_node: which note should contain the leak should there be one
    :param scenario_name: name of the scenario, which becomes the name of the directory that holds the data
    :return:
    """

    # Ensure that these variables are never undefined
    leak_start_ix = 0
    leak_end_ix = 0

    config = ConfigParser()
    config.read('config.ini')

    parameters = ConfigParser()
    parameters['producer'] = config['producer']

    simulation_start_time = config.get('producer', 'simulation_start_time')
    train_start = config.get('producer', 'train_start')
    train_end = config.get('producer', 'train_end')
    val_start = config.get('producer', 'val_start')
    val_end = config.get('producer', 'val_end')
    test_start = config.get('producer', 'test_start')
    test_end = config.get('producer', 'test_end')
    leak_diameter = config.getfloat('producer', 'leak_diameter')
    skip_nodes = json.loads(config.get('producer', 'skip_nodes'))

    # We load an input file here which sets many things up already, but most importantly the demand pattern
    demand_inp_file_path = config.get('producer', 'demand_input_file_path')

    wn = wntr.network.WaterNetworkModel(demand_inp_file_path)

    if is_leak_scenario and leak_node:
        # Compute leak area using area of a circle formula
        leak_area = 3.14159 * (leak_diameter / 2) ** 2

        leak_start_time, leak_end_time, leak_start_datetime, leak_start_ix, leak_end_ix = generate_random_leak_time(
            simulation_start_time,
            test_start,
            test_end,
            wn.options.time.hydraulic_timestep
        )

        node = wn.get_node(leak_node)
        node.add_leak(wn, area=leak_area, start_time=leak_start_time, end_time=leak_end_time)

        parameters['leak_info'] = {}
        parameters['leak_info']['start_time'] = str(leak_start_time)
        parameters['leak_info']['end_time'] = str(leak_end_time)
        parameters['leak_info']['start_datetime'] = leak_start_datetime.to_pydatetime()[0].strftime("%Y-%m-%d %H:%M:%S")
        parameters['leak_info']['leak_area'] = str(leak_area)
        parameters['leak_info']['leak_diameter'] = str(leak_diameter)

    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    timestamps = pd.date_range(
        start=simulation_start_time,
        periods=(wn.options.time.duration / wn.options.time.hydraulic_timestep) + 1,  # wntr generates 1 extra timestamp
        freq=f'{wn.options.time.hydraulic_timestep}s'
    )

    # Add datetime format timestamps to the data since it is better for clarity
    timestamps_df = timestamps.to_frame()
    timestamps_df['ix'] = range(0, len(timestamps_df))

    pressure = sort_columns(results.node['pressure'])
    pressure.index = timestamps

    pressure.drop(columns=skip_nodes, inplace=True)

    flow = sort_columns(results.link['flowrate']) * 3600  # Multiply by 3600 for hourly flow rate
    flow.index = timestamps

    # Split that data into train, validation and test sets
    train_pressure, val_pressure, test_pressure = train_val_test_split(
        pressure,
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end
    )

    train_flow, val_flow, test_flow = train_val_test_split(
        flow,
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end
    )

    # Save the generated data
    if is_leak_scenario and leak_node:
        save_scenario(
            f'{DATASET_DIR}/leak_scenario/{scenario_name}',
            train_pressure,
            val_pressure,
            test_pressure,
            train_flow,
            val_flow,
            test_flow
        )
        test_start_ix = timestamps_df.loc[test_start]['ix']
        test_end_ix = timestamps_df.loc[test_end]['ix']

        # Add +1 to the range here since we want to include the time at index test_end_ix
        labels = [leak_start_ix <= c < leak_end_ix for c in range(test_start_ix, test_end_ix+1)]
        labels_timestamps = pd.date_range(start=test_start, end=test_end, freq=f'{wn.options.time.hydraulic_timestep}s')

        labels_series = pd.Series(
            labels, index=labels_timestamps)

        # Save labels for leak scenario
        labels_series.to_csv(f'{DATASET_DIR}/leak_scenario/{scenario_name}/labels.csv')

        with open(f'{DATASET_DIR}/leak_scenario/{scenario_name}/parameters.ini', 'w') as f:
            # Save used parameters
            parameters.write(f)
    else:
        save_scenario(
            f'{DATASET_DIR}/regular_scenario/{scenario_name}',
            train_pressure,
            val_pressure,
            test_pressure,
            train_flow,
            val_flow,
            test_flow
        )

        with open(f'{DATASET_DIR}/regular_scenario/{scenario_name}/parameters.ini', 'w') as f:
            # Save used parameters
            parameters.write(f)


if __name__ == '__main__':
    config = ConfigParser()
    config.read('config.ini')

    n_scenarios = config.getint('data_generator', 'n_scenarios')
    leak_node = config.get('data_generator', 'leak_node')
    is_leak_scenario = config.getboolean('data_generator', 'is_leak_scenario')

    for scenario_n in range(0, n_scenarios):
        generate_scenario(
            is_leak_scenario=is_leak_scenario,
            leak_node=leak_node,
            scenario_name=f'scenario-{scenario_n}'
        )
