import pickle
import wntr
import matplotlib.pyplot as plt


def train_val_test_split(data, train_i, val_i):
    train = data.loc[:train_i - 1, :]
    val = data.loc[train_i:val_i - 1, :]
    test = data.loc[val_i:, :]
    return train, val, test


def train_val_test_save(folder_path, train, val, test):
    with open(f'{folder_path}train.pkl', 'wb') as f:
        pickle.dump(train, f)

    with open(f'{folder_path}val.pkl', 'wb') as f:
        pickle.dump(val, f)

    with open(f'{folder_path}test.pkl', 'wb') as f:
        pickle.dump(test, f)


def sort_columns(data):
    data.columns = data.columns.astype('int64')
    data = data.reindex(sorted(data.columns), axis=1)
    return data


def run():
    inp_file = 'leakdb/Scenario-13/Hanoi_CMH_Scenario-13.inp'
    wn = wntr.network.WaterNetworkModel(inp_file)
    wn.options.time.duration = 24 * 365 * 3600  # One year in seconds

    sim = wntr.sim.WNTRSimulator(wn)
    results = sim.run_sim()

    pressure = sort_columns(results.node['pressure'].reset_index(drop=True))
    flow = sort_columns(results.link['flowrate'].reset_index(drop=True)) * 3600  # Multiply by 3600 as in LeakDB

    train_i = 24 * 245 * 2
    val_i = 24 * 305 * 2  # Two months for test, two months for val

    pressure_train, pressure_val, pressure_test = train_val_test_split(pressure, train_i, val_i)
    flow_train, flow_val, flow_test = train_val_test_split(flow, train_i, val_i)

    train_val_test_save('synth_dataset/train/pressure/', pressure_train, pressure_val, pressure_test)
    train_val_test_save('synth_dataset/train/flow/', flow_train, flow_val, flow_test)

    plt.plot(results.link['flowrate'].loc[:, '21'] * 3600)
    plt.show()


if __name__ == '__main__':
    run()
