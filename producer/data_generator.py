import json
import os.path
import wntr
from configparser import ConfigParser
import pandas as pd
import numpy as np
import re
import glob


def read_leakdb_scenario_as_dataframe(folder_path, network_property):
    network_data = pd.DataFrame()

    if network_property == 'Pressures':
        file_name = 'Node_*.csv'
    elif network_property == 'Flows':
        file_name = 'Link_*.csv'
    else:
        raise Exception('Incorrect property')

    data_path = os.path.join(folder_path, network_property, file_name)

    print(data_path)

    for d in glob.glob(data_path):
        node_pressure = pd.read_csv(d)['Value']
        node_n = int(re.sub('\D', '', os.path.basename(d)))
        network_data[node_n] = node_pressure
    return network_data.reindex(sorted(network_data.columns), axis=1)


def train_val_test_split(data, train_start, train_end, val_start, val_end, test_start, test_end):
    train = data[train_start:train_end]
    val = data[val_start:val_end]
    test = data[test_start:test_end]
    return train, val, test


def train_val_test_to_csv(folder_path, train, val, test):
    train.to_csv(os.path.join(folder_path, 'train.csv'))
    val.to_csv(os.path.join(folder_path, 'val.csv'))
    test.to_csv(os.path.join(folder_path, 'test.csv'))


def save_scenario(folder_path, train_pressure, val_pressure, test_pressure, train_flow, val_flow, test_flow):
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
    data.columns = data.columns.astype('int64')
    data = data.reindex(sorted(data.columns), axis=1)
    return data


def generate_random_leak_time(simulation_start_time, test_start, test_end, hydraulic_timestep):
    # Note that this includes the first point of the test set, but this is good since the leak should not start at t=1
    timestamps = pd.date_range(start=simulation_start_time, end=test_end, freq=f'{hydraulic_timestep}s')
    timestamps_df = timestamps.to_frame()
    timestamps_df['ix'] = range(0, len(timestamps_df))
    possible_leak_times = timestamps_df[test_start:test_end]
    leak_start = possible_leak_times.sample()
    leak_start_ix = leak_start['ix'].iloc[0]
    leak_start_datetime = leak_start.index
    leak_end_ix = np.random.randint(low=leak_start_ix, high=len(timestamps_df))

    leak_start_time = leak_start_ix * hydraulic_timestep
    leak_end_time = leak_end_ix * hydraulic_timestep

    return leak_start_time, leak_end_time, leak_start_datetime, leak_start_ix, leak_end_ix


def generate_scenario(is_leak_scenario=False, leak_node=None, scenario_name='scenario-2'):
    leak_start_ix = 0
    leak_end_ix = 0

    config = ConfigParser()
    config.read('../config.ini')

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

    wn = wntr.network.WaterNetworkModel('../' + demand_inp_file_path)

    if is_leak_scenario and leak_node:
        leak_diameter = leak_diameter
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

    timestamps_df = timestamps.to_frame()
    timestamps_df['ix'] = range(0, len(timestamps_df))

    pressure = sort_columns(results.node['pressure'])
    pressure.index = timestamps

    pressure.drop(columns=skip_nodes, inplace=True)

    flow = sort_columns(results.link['flowrate']) * 3600  # Multiply by 3600 for hourly flow rate
    flow.index = timestamps

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

    if is_leak_scenario and leak_node:
        save_scenario(
            f'../dataset/leak_scenario/{scenario_name}',
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
        labels_series.to_csv(f'../dataset/leak_scenario/{scenario_name}/labels.csv')
    else:
        save_scenario(
            f'../dataset/regular_scenario/{scenario_name}',
            train_pressure,
            val_pressure,
            test_pressure,
            train_flow,
            val_flow,
            test_flow
        )

    with open(f'../dataset/leak_scenario/{scenario_name}/parameters.ini', 'w') as f:
        parameters.write(f)


if __name__ == '__main__':
    config = ConfigParser()
    config.read('../config.ini')

    generate_scenario(
        is_leak_scenario=True,
        leak_node='2',
        scenario_name='scenario-2'
    )
