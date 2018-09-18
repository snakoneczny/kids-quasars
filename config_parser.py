import yaml

from utils import FEATURES
from models import MODEL_CONSTRUCTORS


def parse_config(config_file):
    with open(config_file, 'r') as config_file:
        cfg = yaml.load(config_file)

    cfg = add_experiment_name(cfg)

    # Get features from a general feature subset name
    cfg['features'] = FEATURES[cfg['features']]
    cfg['n_features'] = len(cfg['features'])

    model_constructor = MODEL_CONSTRUCTORS[cfg['model']]

    return model_constructor, cfg


def add_experiment_name(cfg):
    cfg['exp_name'] = '{model}_f-{features}_cut-{cut}'.format(
        model=cfg['model'], features=cfg['features'], cut=cfg['cut'])

    if cfg['clean_sdss']:
        cfg['exp_name'] = 'sdss-clean_{exp_name}'.format(exp_name=cfg['exp_name'])

    data_name = cfg['data_name'].replace('.', '_')

    cfg['exp_name'] = '{data_name}_{exp_name}'.format(data_name=data_name, exp_name=cfg['exp_name'])

    return cfg
