import matplotlib.pyplot as plt
from epynet import Network
import csv


def set_config():
    # TODO: set simulation time here
    # It seems that the simulation time has to be set in the .inp file when using epynet
    print('To be implemented')


def main():
    """
    This method now runs a quick example simulation to show how we can generate a dataset using the Net3.inp sample
    network from WNTR. It shows a plot of the progression of pressure in one of the nodes throughout the simulation.
    At the end some sample epynet_data is exported to a csv.
    :return:
    """
    network = Network('network_input_files/Hanoi_CMH.inp')
    network.run()

    pressure_node_2 = network.nodes['2'].pressure
    pressure_node_2.plot(color='r')
    plt.xlabel('time (seconds)')
    plt.ylabel('pressure (psi)')
    plt.title('Pressure of node 2: EPYNET-Vitens')
    plt.show()

    flow_pipe_20 = network.pipes['20'].flow

    with open('dataset/output_data/epynet_sim_data.csv', 'w') as sim_data:
        writer = csv.writer(sim_data)
        writer.writerow(['pressure_node_2', 'flow_pipe_20'])
        for (p, f) in zip(pressure_node_2, flow_pipe_20):
            writer.writerow([p, f])


if __name__ == '__main__':
    main()
