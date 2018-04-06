import yaml

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils import FEATURES

MODELS = {
    'rf': RandomForestClassifier(n_estimators=400, random_state=491237, criterion='entropy', n_jobs=8),
    'xgb': XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=200, objective='multi:softmax', booster='gbtree',
                         n_jobs=8, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                         colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,
                         random_state=18235, missing=None)
}


def parse_config(config_file):
    with open(config_file, 'r') as config_file:
        cfg = yaml.load(config_file)

    cfg = add_experiment_name(cfg)

    cfg['features'] = FEATURES[cfg['features']]

    model = MODELS[cfg['model']]

    return model, cfg


def add_experiment_name(cfg):
    cfg['exp_name'] = '{model}_f-{features}_cut-{cut}'.format(
        model=cfg['model'], features=cfg['features'], cut=cfg['cut'])

    if cfg['clean_sdss']:
        cfg['exp_name'] = 'sdss-clean_{exp_name}'.format(exp_name=cfg['exp_name'])

    cfg['exp_name'] = '{data_name}_{exp_name}'.format(data_name=cfg['data_name'].replace('.', '_'),
                                                      exp_name=cfg['exp_name'])
    return cfg
