from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical

tfd = tfp.distributions


def build_rf_clf(params):
    n_estimators = 10 if params['is_test'] else 500
    return RandomForestClassifier(
        n_estimators=n_estimators, criterion='gini', random_state=491237, n_jobs=24, verbose=2
    )


def build_rf_reg(params):
    if params['is_test']:
        n_estimators = 10
    elif params['specialization'] == 'QSO':
        n_estimators = 2000
    elif params['specialization'] == 'GALAXY':
        n_estimators = 400
    else:  # All classes
        n_estimators = 1200

    return RandomForestRegressor(
        n_estimators=n_estimators, criterion='mse', random_state=134, n_jobs=24, verbose=2
    )


def build_xgb_clf(params):
    if params['is_test']:
        n_estimators = 10
    elif params['is_inference']:
        if params['features_key'] == 'colors':
            n_estimators = 100
        elif params['features_key'] == 'no-sg':
            n_estimators = 260
        else:  # All features
            n_estimators = 380
    else:
        n_estimators = 100000

    return XGBClassifier(
        max_depth=9, learning_rate=0.1, gamma=0, min_child_weight=1, colsample_bytree=0.9, subsample=0.8,
        scale_pos_weight=2, reg_alpha=0, reg_lambda=1, n_estimators=n_estimators, objective='multi:softmax',
        booster='gbtree', max_delta_step=0, colsample_bylevel=1, base_score=0.5, random_state=18235, missing=None,
        verbosity=0, n_jobs=24,
    )


def build_xgb_reg(params):
    # All classes
    if not params['specialization']:
        return XGBRegressor(
            max_depth=5, learning_rate=0.1, gamma=0, min_child_weight=20, colsample_bytree=0.5, subsample=1,
            scale_pos_weight=1, reg_alpha=1, reg_lambda=1, n_estimators=100000, objective='reg:squarederror',
            booster='gbtree', max_delta_step=0, colsample_bylevel=1, colsample_bynode=1, base_score=0.5,
            random_state=1587, missing=None, importance_type='gain', verbosity=0, n_jobs=24,
        )

    elif params['specialization'] == 'QSO':
        # TODO: subsample danger, new random_state?
        return XGBRegressor(
            max_depth=5, learning_rate=0.1, gamma=0, min_child_weight=10, colsample_bytree=0.6, subsample=0.4,
            scale_pos_weight=1, reg_alpha=0, reg_lambda=1, n_estimators=100000, objective='reg:squarederror',
            booster='gbtree', max_delta_step=0, colsample_bylevel=1, colsample_bynode=1, base_score=0.5,
            random_state=1587, missing=None, importance_type='gain', verbosity=0, n_jobs=24,
        )

    elif params['specialization'] == 'GALAXY':
        return XGBRegressor(
            max_depth=7, learning_rate=0.1, gamma=0, min_child_weight=20, colsample_bytree=1, subsample=1,
            scale_pos_weight=1, reg_alpha=0, reg_lambda=2, n_estimators=100000, objective='reg:squarederror',
            booster='gbtree', max_delta_step=0, colsample_bylevel=1, colsample_bynode=1, base_score=0.5,
            random_state=1587, missing=None, importance_type='gain', verbosity=0, n_jobs=24,
        )

    else:
        raise Exception('Not implemented redshift specialization: {}'.format(params['specialization']))


class AnnClf(BaseEstimator):

    def __init__(self, params):
        self.params_exp = params
        self.network = None
        self.scaler = MinMaxScaler()
        self.patience = 1000
        self.batch_size = 256
        self.lr = 0.0001
        self.dropout_rate = 0.2
        self.metric_names = ['categorical_crossentropy', 'accuracy']

        if params['is_test']:
            self.epochs = 10
        elif params['is_inference']:
            self.epochs = 400
        else:
            self.epochs = 4000

        log_name = 'clf, lr={}, bs={}, {}'.format(self.lr, self.batch_size, params['timestamp_start'].replace('_', ' '))
        if params['tag']:
            log_name = '{}, {}'.format(params['tag'], log_name)

        self.callbacks = []
        if not self.params_exp['is_inference']:
            self.callbacks.append(EarlyStopping(monitor='val_accuracy', patience=self.patience, restore_best_weights=True))
        if not self.params_exp['is_test']:
            self.callbacks.append(CustomTensorBoard(log_folder=log_name, params=self.params_exp,
                                                    is_inference=self.params_exp['is_inference']))

    def __create_network(self, params):
        model = Sequential()
        model.add(Dense(100, input_dim=params['n_features'], activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(40, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(40, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(3, activation='softmax', name='category'))

        opt = Adam(lr=self.lr)

        loss = {'category': 'categorical_crossentropy'}
        model.compile(loss=loss, optimizer=opt, metrics=self.metric_names)

        return model

    def fit(self, X, y, validation_data_arr=None):
        # TODO: extract a function
        X = self.scaler.fit_transform(X)
        y['category'] = to_categorical(y['category'])

        if validation_data_arr:
            validation_data_arr = [(self.scaler.transform(validation_data_arr[i][0]),
                                    {'category': to_categorical(validation_data_arr[i][1]['category'])},
                                    validation_data_arr[i][2]) for i in range(len(validation_data_arr))]

            validation_callback = AdditionalValidationSets(validation_data_arr[1:],
                                                           tracked_metric_names=self.metric_names)
            self.callbacks = [validation_callback] + self.callbacks

            validation_data = (validation_data_arr[0][0], validation_data_arr[0][1])

        else:
            validation_data = None

        self.network = self.__create_network(self.params_exp)
        self.network.fit(X, y, validation_data=validation_data, epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=1)

    def predict(self, X, encoder):
        X = self.scaler.transform(X)
        y_pred_proba = self.network.predict(X)
        predictions_df = decode_clf_preds(y_pred_proba, encoder)
        return predictions_df


class AnnReg(BaseEstimator):

    def __init__(self, params):
        self.params_exp = params
        self.network = None
        self.scaler = MinMaxScaler()
        self.patience = 1000
        self.batch_size = 256
        self.lr = 0.0001
        self.dropout_rate = 0.2

        self.metrics_dict = {
            'mean_squared_error': mean_squared_error,
            'mean_absolute_error': mean_absolute_error,
        }

        if params['is_test']:
            self.epochs = 10
        elif params['is_inference']:
            if params['specialization'] == 'QSO':
                self.epochs = 320
            elif params['specialization'] == 'GALAXY':
                self.epochs = 200
            else:
                self.epochs = 340
        else:
            self.epochs = 5000

        base_name = 'z-{}'.format(params['specialization']).lower() if params['specialization'] else 'z'
        log_name = '{}, lr={}, bs={}, {}'.format(base_name, self.lr, self.batch_size,
                                                 params['timestamp_start'].replace('_', ' '))
        if params['tag']:
            log_name = '{}, {}'.format(params['tag'], log_name)

        self.callbacks = []
        if not (self.params_exp['is_inference'] or self.params_exp['is_test']):
            self.callbacks.append(EarlyStopping(monitor='val_loss', patience=self.patience,
                                                restore_best_weights=True))
            self.callbacks.append(CustomTensorBoard(log_folder=log_name, params=self.params_exp,
                                                    is_inference=self.params_exp['is_inference']))

    def __create_network(self, params):
        model = Sequential()
        model.add(Dense(100, input_dim=params['n_features'], activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(80, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(40, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(40, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(2))
        model.add(tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])),
            name='redshift',
        ))

        opt = Adam(lr=self.lr)

        negloglik = lambda y, p_y: -p_y.log_prob(y)
        loss = {'redshift': negloglik}

        model.compile(loss=loss, optimizer=opt)

        return model

    def fit(self, X, y, validation_data_arr=None):
        # TODO: extract a function, differs from clf network only with respect to y
        X = self.scaler.fit_transform(X)

        if validation_data_arr:
            validation_data_arr = [(self.scaler.transform(validation_data_arr[i][0]),
                                    validation_data_arr[i][1],
                                    validation_data_arr[i][2]) for i in range(len(validation_data_arr))]

            validation_callback = AdditionalValidationSets(
                validation_data_arr, additional_metrics_dict=self.metrics_dict, model_wrapper=self)
            self.callbacks = [validation_callback] + self.callbacks

            validation_data = (validation_data_arr[0][0], validation_data_arr[0][1])

        else:
            validation_data = None

        self.network = self.__create_network(self.params_exp)
        self.network.fit(X, y, validation_data=validation_data, epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=1)

    def predict(self, X, encoder=None, scale_data=True):
        X_to_pred = self.scaler.transform(X) if scale_data else X
        predictions_df = pd.DataFrame()
        preds = self.network(X_to_pred)
        predictions_df['Z_PHOTO'] = preds.mean()[:, 0]
        predictions_df['Z_PHOTO_STDDEV'] = preds.stddev()[:, 0]
        return predictions_df


class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets, tracked_metric_names=None, additional_metrics_dict=None, model_wrapper=None,
                 verbose=0, batch_size=None):
        """
        :param validation_sets:
        a list of 3-tuples (validation_data, validation_targets, validation_set_name)
        or 4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        """
        super(AdditionalValidationSets, self).__init__()
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3]:
                raise ValueError()
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size
        self.tracked_metric_names = tracked_metric_names
        self.additional_metrics_dict = additional_metrics_dict
        self.model_wrapper = model_wrapper

    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)

        # Record the same values as History() as well
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # Evaluate on the additional validation sets
        for validation_set in self.validation_sets:
            if len(validation_set) == 3:
                validation_data, validation_targets, validation_set_name = validation_set
                sample_weights = None
            elif len(validation_set) == 4:
                validation_data, validation_targets, sample_weights, validation_set_name = validation_set
            else:
                raise ValueError()

            # First, standard model evaluation on tracked metrics
            results = self.model.evaluate(x=validation_data, y=validation_targets, verbose=self.verbose,
                                          sample_weight=sample_weights, batch_size=self.batch_size)
            if not isinstance(results, Iterable):
                results = [results]
            for i, result in enumerate(results):
                if i == 0:
                    value_name = 'val_' + validation_set_name + '_loss'
                else:
                    value_name = 'val_' + validation_set_name + '_' + self.tracked_metric_names[i - 1]
                self.history.setdefault(value_name, []).append(result)
                logs[value_name] = result

            # Second, additional metrics with custom predict function from model wrapper
            if self.additional_metrics_dict:

                # TODO: should not assume redshift (though it's never going to be anything else?)
                preds = self.model_wrapper.predict(validation_data, scale_data=False)
                y_pred = preds['Z_PHOTO']

                # print('\n{}: z std mean: {}'.format(validation_set_name, preds['Z_PHOTO_STDDEV'].mean()))
                # print('{}: z std std: {}'.format(validation_set_name, preds['Z_PHOTO_STDDEV'].std()))

                for metric_name, metric_func in self.additional_metrics_dict.items():
                    result = metric_func(validation_targets['redshift'], y_pred)
                    value_name = 'val_{}_{}'.format(validation_set_name, metric_name)
                    self.history.setdefault(value_name, []).append(result)
                    logs[value_name] = result

        super().on_epoch_end(epoch, logs)


class CustomTensorBoard(TensorBoard):
    def __init__(self, log_folder, params, is_inference):
        subfolder = 'inf' if is_inference else 'exp'
        log_dir = './outputs/tensorboard/{}/{}'.format(subfolder, log_folder)
        super().__init__(log_dir=log_dir, profile_batch=0)

        self.params_exp = params
        self.main_test_name = 'top' if params['test_method'] == 'magnitude' else 'random'

    def on_epoch_end(self, epoch, logs=None):
        logs_to_send = logs.copy()

        # Add learning rate
        logs_to_send.update({'learning rate': K.eval(self.model.optimizer.lr)})

        # Remove losses tracked for gradient descent
        # to_pop = []  # ['loss', 'random_loss']
        # if self.params_exp['pred_class'] and self.params_exp['pred_z']:
        #     to_pop += [
        #         # Losses are not needed as we track all metrics with their proper names
        #         'category_loss',
        #         'redshift_loss',
        #         # Below metrics are artificially created due to metrics tracking
        #         'redshift_categorical_crossentropy',
        #         'redshift_acc',
        #         # The same for category
        #         'category_mean_squared_error',
        #         'category_mean_absolute_error',
        #     ]
        # for str in to_pop:
        #     logs_to_send.pop(str, None)
        #     logs_to_send.pop('{}_{}'.format(self.main_test_name, str), None)

        # Standalone problems, add category or redshift to metric names
        # if not (self.params_exp['pred_class'] and self.params_exp['pred_z']):
        #     metrics = ['categorical_crossentropy', 'acc'] if self.params_exp['pred_class'] \
        #         else ['mean_squared_error', 'mean_absolute_error']
        #     for metric in metrics:
        #         logs_to_send['train_{}'.format(metric)] = logs_to_send.pop('{}'.format(metric))
        #         logs_to_send['{}_{}'.format(self.main_test_name, metric)] = logs_to_send.pop(
        #             'val_{}'.format(metric))
        # TODO: add to alll metrics exp name: t = 'category' if self.params_exp['pred_class'] else 'redshift'

        # Check if additional random validation sets present and put val_random on the same plot with train
        if any([log_name.startswith('val_random') for log_name in logs_to_send]):
            for key in list(logs_to_send):
                # TODO: if val_loss != val_random_loss to val_random_loss -> val_loss
                if key.startswith('val_random'):
                    key_1 = key.replace('val_random', 'val')
                    key_2 = key.replace('val_random', 'val_' + self.main_test_name)
                    # Move loss used in training to it's proper name if not already done in validation
                    if key_1 in logs_to_send:
                        logs_to_send[key_2] = logs_to_send[key_1]
                    # Move random score in order to plot with training scores
                    logs_to_send[key_1] = logs_to_send[key]

        super().on_epoch_end(epoch, logs_to_send)


# TODO: this function transforms y to categorical and scales X, could it be merged with build_validation_data?
def create_multioutput_data(y, validation_data, scaler):
    y_categorical = to_categorical(y['category'])
    y['category'] = y_categorical

    y_val_categorical = to_categorical(validation_data[1]['category'])
    validation_data = (scaler.transform(validation_data[0]),
                       {'category': y_val_categorical, 'redshift': validation_data[1]['redshift']})

    return y, validation_data


# TODO: this should be in a pipeline for non ann models, also encoder usage is multiplied in networks and here
def get_single_problem_predictions(model, X, encoder, cfg):
    params_pred = {'ntree_limit': model.best_ntree_limit} if hasattr(model, 'best_ntree_limit') else {}
    if cfg['pred_class']:
        y_pred_proba = model.predict_proba(X, **params_pred)
        predictions = decode_clf_preds(y_pred_proba, encoder)
    else:
        predictions = pd.DataFrame()
        predictions['Z_PHOTO'] = model.predict(X, **params_pred)
    return predictions


def decode_clf_preds(y_pred_proba, encoder):
    predictions_df = pd.DataFrame()
    for i, c in enumerate(encoder.classes_):
        predictions_df['{}_PHOTO'.format(c)] = y_pred_proba[:, i]
    predictions_df['CLASS_PHOTO'] = encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))
    return predictions_df


def get_model(cfg):
    constructors = {
        'rf-class': build_rf_clf,
        'rf-redshift': build_rf_reg,
        'xgb-class': build_xgb_clf,
        'xgb-redshift': build_xgb_reg,
        'ann-class': AnnClf,
        'ann-redshift': AnnReg,
    }

    name = cfg['model']
    if cfg['pred_class']:
        name += '-class'
    if cfg['pred_z']:
        name += '-redshift'

    model = constructors[name](cfg)
    return model


def build_ann_output_dict(y, z, cfg):
    outputs = {}
    if cfg['pred_class']:
        outputs['category'] = y
    if cfg['pred_z']:
        outputs['redshift'] = z
    return outputs


def build_outputs(y, z, cfg):
    if cfg['model'] == 'ann':
        outputs = build_ann_output_dict(y, z, cfg)
    else:
        if cfg['pred_class']:
            outputs = y
        else:
            outputs = z
    return outputs


def build_ann_validation_data(X_val_arr, y_val_arr, z_val_arr, test_names_arr, cfg):
    validation_data = [(X_val_arr[i], build_ann_output_dict(y_val_arr[i], z_val_arr[i], cfg), test_names_arr[i]) for i
                       in range(len(X_val_arr))]
    return validation_data


def build_xgb_validation_data(X_val_arr, y_val_arr, z_val_arr, cfg):
    if cfg['pred_class']:
        val_list = [(X_val_arr[i], y_val_arr[i]) for i in range(len(X_val_arr))]
    else:
        val_list = [(X_val_arr[i], z_val_arr[i]) for i in range(len(X_val_arr))]
    # Reverse is needed as early stopping uses the last eval set
    val_list.reverse()
    return val_list
