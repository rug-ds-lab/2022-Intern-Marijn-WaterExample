# Leak Detection ECiDA-Vitens
As of now the repository contains two simple scripts demonstrating how the data generation for the project could be 
approached. These are very basic at the moment and just serve as an example.

## Instructions
Firstly, install the project requirements using the following command:
```shell
pip install -r requirements.txt
```
# Vitens-epynet
To run the script `epynet_run_sym.py` first the [epynet](https://github.com/Vitens/epynet) tool needs to be installed.
This can be done using the instructions at the [repository](https://github.com/Vitens/epynet) or by running the shell 
script `epynet-install.sh`. Then we can simply run:

```shell
python3 epynet_run_sim.py
```

# WNTR
The [WNTR](https://wntr.readthedocs.io/en/latest/) tool is installed alongside the requirements so we can simply run:
```shell
python3 wntr_run_sim.py
```