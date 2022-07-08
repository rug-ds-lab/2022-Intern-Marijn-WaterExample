import wntr
import matplotlib.pyplot as plt
from configparser import ConfigParser
import csv


def set_config(wn):
    """
    A method for setting a number of parameters for the WaterNetworkModel. For now, it is only covers some values to
    control simulation time. Later, this will be extended to all the needed parameters.
    :param wn: WaterNetworkModel object
    :return:
    """
    config = ConfigParser()
    config.read('config.ini')
    wn.options.time.duration = int(config['DEFAULT']['duration'])
    wn.options.time.hydraulic_timestep = int(config['DEFAULT']['hydraulic_timestep'])
    wn.options.time.quality_timestep = int(config['DEFAULT']['quality_timestep'])
    wn.options.time.rule_timestep = float(config['DEFAULT']['rule_timestep'])
    wn.options.time.pattern_timestep = int(config['DEFAULT']['pattern_timestep'])


def main():
    """
    This method now runs a quick example simulation to show how we can generate a dataset using the Net3.inp sample
    network from WNTR. It shows a plot of the pressure in the network at the end of the simulation, and a plot of the
    progression of pressure in one of the nodes throughout the simulation. At the end some sample epynet_data is exported to
    a csv.
    :return:
    """
    wn = wntr.network.WaterNetworkModel('dataset/network_input_files/Hanoi_CMH.inp')
    wntr.graphics.plot_network(wn, title=wn.name)
    plt.show()

    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()

    pressure_at_5hr = results.node['pressure'].loc[wn.options.time.duration, :]
    wntr.graphics.plot_network(wn, node_attribute=pressure_at_5hr, node_size=30,
                               title=f'Pressure at t={wn.options.time.duration}')
    plt.show()

    pressure = results.node['pressure']
    pressure_node_2 = pressure.loc[:, '2']
    pressure_node_2.plot()

    flow_pipe_20 = results.link['flowrate'].loc[:, '20']

    plt.xlabel('time (seconds)')
    plt.ylabel('pressure (psi)')
    plt.title('Pressure of node 2: WNTR')
    plt.show()

    with open('dataset/output_data/wntr_sim_data.csv', 'w') as sim_data:
        writer = csv.writer(sim_data)
        writer.writerow(['pressure_node_2', 'flow_pipe_20'])
        for (p, f) in zip(pressure_node_2, flow_pipe_20):
            writer.writerow([p, f])


if __name__ == '__main__':
    main()
