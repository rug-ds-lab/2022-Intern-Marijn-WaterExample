import pickle
from configparser import ConfigParser


def load_model():
    config = ConfigParser()
    config.read('config.ini')

    file_name = config.get('pipeline1', 'model_file_name')

    with open(f'model/{file_name}', 'rb') as f:
        random_forest_clf = pickle.load(f)
    return random_forest_clf
