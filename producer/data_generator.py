import os.path

import matplotlib.pyplot as plt
import wntr
from configparser import ConfigParser
import pandas as pd
import numpy as np


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
    leak_start_ix = possible_leak_times.sample()['ix'].iloc[0]
    leak_end_ix = np.random.randint(low=leak_start_ix, high=len(timestamps_df))

    leak_start_time = leak_start_ix * hydraulic_timestep
    leak_end_time = leak_end_ix * hydraulic_timestep

    return leak_start_time, leak_end_time


def generate_scenario(is_leak_scenario=False, node_name=None, scenario_name='scenario'):
    config = ConfigParser()
    config.read('../config.ini')

    simulation_start_time = config.get('producer', 'simulation_start_time')
    train_start = config.get('producer', 'train_start')
    train_end = config.get('producer', 'train_end')
    val_start = config.get('producer', 'val_start')
    val_end = config.get('producer', 'val_end')
    test_start = config.get('producer', 'test_start')
    test_end = config.get('producer', 'test_end')

    # We load an input file here which sets many things up already, but most importantly the demand pattern
    demand_inp_file_path = config.get('producer', 'demand_input_file_path')
    print(demand_inp_file_path)

    wn = wntr.network.WaterNetworkModel('../' + demand_inp_file_path)

    if is_leak_scenario:
        leak_diameter = np.random.uniform(0.02, 0.2)
        leak_area = 3.14159 * (leak_diameter / 2) ** 2

        leak_start_time, leak_end_time = generate_random_leak_time(
            simulation_start_time,
            test_start,
            test_end,
            wn.options.time.hydraulic_timestep
        )

        if node_name:
            node = wn.get_node(node_name)
            node.add_leak(wn, area=leak_area, start_time=leak_start_time, end_time=leak_end_time)

    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    timestamps = pd.date_range(
        start=simulation_start_time,
        periods=(wn.options.time.duration / wn.options.time.hydraulic_timestep) + 1,  # wntr generates 1 extra timestamp
        freq=f'{wn.options.time.hydraulic_timestep}s'
    )

    pressure = sort_columns(results.node['pressure'])
    pressure.index = timestamps

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

    print(train_pressure)
    print(val_pressure)
    print(test_pressure)

    if is_leak_scenario:
        save_scenario(
            f'dataset/leak_scenario/{scenario_name}',
            train_pressure,
            val_pressure,
            test_pressure,
            train_flow,
            train_pressure,
            test_pressure
        )
    else:
        save_scenario(
            f'dataset/regular_scenario/{scenario_name}',
            train_pressure,
            val_pressure,
            test_pressure,
            train_flow,
            train_pressure,
            test_pressure
        )

    plt.plot(test_pressure[2])
    plt.show()


if __name__ == '__main__':
    generate_scenario(
        is_leak_scenario=True,
        node_name='2',
        scenario_name='scenario'
    )
