import yaml
import datetime

from data import FEATURES


def get_config(args, is_inference=False):
    cfg = vars(args).copy()

    with open(cfg['config_file'], 'r') as config_file:
        cfg.update(yaml.full_load(config_file))

    cfg['is_inference'] = is_inference

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
    if cfg['pred_class'] and cfg['pred_z']:
        problem = 'clf-z'
    elif cfg['pred_class']:
        problem = 'clf'
    else:
        problem = 'z'

    cfg['exp_name'] = '{model}_{problem}_f-{features}'.format(
        model=cfg['model'], problem=problem, features=cfg['features_key'])

    # if not cfg['is_inference']:
    #     cfg['exp_name'] += '_test-{test}'.format(test=cfg['test_method'])

    if cfg['specialization']:
        cfg['exp_name'] += '_spec-{}'.format(cfg['specialization'].lower())

    train_data = cfg['train_data'].replace('.', '_')
    cfg['exp_name'] = '{train_data}_{exp_name}'.format(train_data=train_data, exp_name=cfg['exp_name'])

    return cfg
