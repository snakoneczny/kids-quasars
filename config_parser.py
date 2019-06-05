import yaml
import datetime

from data import FEATURES


def get_config(args):
    cfg = vars(args).copy()

    with open(cfg['config_file'], 'r') as config_file:
        cfg.update(yaml.load(config_file))

    # Add experiment start timestamp
    cfg['timestamp_start'] = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    # Make specialization subset upper case to match table values
    if cfg['specialization']:
        cfg['specialization'] = cfg['specialization'].upper()

    # Get features from a general feature subset name
    cfg['features'] = FEATURES[cfg['features_key']]
    cfg['n_features'] = len(cfg['features'])

    cfg = add_experiment_name(cfg)

    return cfg


def add_experiment_name(cfg):
    cfg['exp_name'] = '{model}_f-{features}_test-{test}'.format(
        model=cfg['model'], features=cfg['features_key'], test=cfg['test'])

    if cfg['specialization']:
        cfg['exp_name'] += 'spec-{}'.format(cfg['specialization'])

    train_data = cfg['train_data'].replace('.', '_')
    cfg['exp_name'] = '{train_data}_{exp_name}'.format(train_data=train_data, exp_name=cfg['exp_name'])

    return cfg
