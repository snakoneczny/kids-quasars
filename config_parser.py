import yaml
import datetime

from data import FEATURES


def get_config(args):
    cfg = vars(args).copy()

    with open(cfg['config_file'], 'r') as config_file:
        cfg.update(yaml.load(config_file))

    cfg = add_experiment_name(cfg)
    cfg['timestamp_start'] = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    # Get features from a general feature subset name
    cfg['features'] = FEATURES[cfg['features']]
    cfg['n_features'] = len(cfg['features'])

    return cfg


def add_experiment_name(cfg):
    cfg['exp_name'] = '{model}_f-{features}_test-{test}'.format(
        model=cfg['model'], features=cfg['features'], test=cfg['test'])

    train_data = cfg['train_data'].replace('.', '_')

    cfg['exp_name'] = '{train_data}_{exp_name}'.format(train_data=train_data, exp_name=cfg['exp_name'])

    return cfg
