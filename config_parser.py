import yaml

from data import FEATURES
from models import get_model_constructor


def parse_config(config_file):
    with open(config_file, 'r') as config_file:
        cfg = yaml.load(config_file)

    cfg = add_experiment_name(cfg)

    # Get features from a general feature subset name
    cfg['features'] = FEATURES[cfg['features']]
    cfg['n_features'] = len(cfg['features'])

    model_constructor = get_model_constructor(cfg)

    return model_constructor, cfg


def add_experiment_name(cfg):
    cfg['exp_name'] = '{model}_f-{features}_test-{test}'.format(
        model=cfg['model'], features=cfg['features'], test=cfg['test'])

    train_data = cfg['train_data'].replace('.', '_')

    cfg['exp_name'] = '{train_data}_{exp_name}'.format(train_data=train_data, exp_name=cfg['exp_name'])

    return cfg
