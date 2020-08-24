import os
from collections.abc import Iterable
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from env_config import PROJECT_PATH
from utils import plot_to_image
from evaluation import redshift_scatter_plot
from plotting import plot_confusion_matrix

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
    elif params['is_inference']:  # Not up-to-date
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
            random_state=273453, missing=None, importance_type='gain', verbosity=0, n_jobs=24,
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


# TODO: extract common parts with ANN regression
# TODO: move encoder to the network
class AnnClf(BaseEstimator):

    def __init__(self, params):
        self.params_exp = params
        self.network = None
        self.scaler = MinMaxScaler()
        self.patience = 10000
        self.batch_size = 512
        self.lr = 0.0001
        self.dropout_rate = 0.05
        self.metric_names = ['accuracy']
        self.model_path = 'outputs/inf_models/{exp_name}__{timestamp}.hdf5'.format(
            exp_name=params['exp_name'], timestamp=params['timestamp_start'])
        self.callbacks = []
        self.tensorboard_callback = None

        if params['is_test']:
            self.epochs = 10
        elif params['is_inference']:
            self.epochs = 350
        else:
            self.epochs = 10000

        log_name = 'clf, lr={}, bs={}, {}'.format(self.lr, self.batch_size, params['timestamp_start'].replace('_', ' '))
        if params['tag']:
            log_name = '{}, {}'.format(params['tag'], log_name)

        self.callbacks = []
        if self.params_exp['is_inference']:
            self.callbacks.append(ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True,
                                                  save_weights_only=True, verbose=1))
        else:
            self.callbacks.append(EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True))

        if not self.params_exp['is_test']:
            self.tensorboard_callback = CustomTensorBoard(log_folder=log_name, params=self.params_exp,
                                                          is_inference=self.params_exp['is_inference'])
            self.callbacks.append(self.tensorboard_callback)

    def create_network(self, params):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=params['n_features'],
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(128, activation='relu',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dropout(self.dropout_rate))

        model.add(Dense(3, activation='softmax', name='category',
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5)
                        ))

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
            validation_callback = AdditionalValidationSets(
                validation_data_arr, self.params_exp, model_wrapper=self, tracked_metric_names=self.metric_names,
                tensorboard_callback=self.tensorboard_callback
            )
            self.callbacks = [validation_callback] + self.callbacks
            validation_data = (validation_data_arr[0][0], validation_data_arr[0][1])
        else:
            validation_data = None

        self.network = self.create_network(self.params_exp)
        self.network.fit(X, y, validation_data=validation_data, epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=1)

        # Restore best weights if inference
        if self.params_exp['is_inference']:
            self.network.load_weights(self.model_path)

    def predict(self, X, encoder=None, scale_data=True, batch_size=None):
        X_to_pred = self.scaler.transform(X) if scale_data else X
        y_pred_proba = self.network.predict(X_to_pred)
        predictions_df = decode_clf_preds(y_pred_proba, encoder)
        return predictions_df


class AnnReg(BaseEstimator):

    def __init__(self, params):
        self.params_exp = params
        self.network = None
        self.scaler = MinMaxScaler()
        self.patience = 150
        self.batch_size = 256
        self.lr = 0.0001
        self.callbacks = []
        self.tensorboard_callback = None

        self.model_path = 'outputs/inf_models/{exp_name}__{timestamp}.hdf5'.format(
            exp_name=params['exp_name'], timestamp=params['timestamp_start'])

        self.metrics_dict = {
            'root_mean_squared_error': partial(mean_squared_error, squared=False),
            'mean_absolute_error': mean_absolute_error,
        }

        if params['is_test']:
            self.epochs = 10
        elif params['is_inference']:
            if params['specialization'] == 'QSO':
                self.epochs = 200
            elif params['specialization'] == 'GALAXY':
                self.epochs = 500
            else:
                self.epochs = 1000
        else:
            self.epochs = 20000

        base_name = 'z-{}'.format(params['specialization']).lower() if params['specialization'] else 'z'
        log_name = '{}, lr={}, bs={}, {}'.format(base_name, self.lr, self.batch_size,
                                                 params['timestamp_start'].replace('_', ' '))
        if params['tag']:
            log_name = '{}, {}'.format(params['tag'], log_name)

        if self.params_exp['is_inference']:
            self.callbacks.append(ModelCheckpoint(self.model_path, monitor='val_loss', save_best_only=True,
                                                  save_weights_only=True, verbose=1))
        else:
            self.callbacks.append(EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True))
        if not self.params_exp['is_test']:
            self.tensorboard_callback = CustomTensorBoard(log_folder=log_name, params=self.params_exp,
                                                          is_inference=self.params_exp['is_inference'])
            self.callbacks.append(self.tensorboard_callback)

    def create_network(self, params):
        model = Sequential()
        model.add(Dense(512, kernel_initializer='normal', activation='relu', input_dim=params['n_features'],
                        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        # bias_regularizer=regularizers.l2(1e-4),
                        # activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dense(512, kernel_initializer='normal', activation='relu',
                        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        # bias_regularizer=regularizers.l2(1e-4),
                        # activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dense(512, kernel_initializer='normal', activation='relu',
                        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        # bias_regularizer=regularizers.l2(1e-4),
                        # activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dense(512, kernel_initializer='normal', activation='relu',
                        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        # bias_regularizer=regularizers.l2(1e-4),
                        # activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dense(512, kernel_initializer='normal', activation='relu',
                        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        # bias_regularizer=regularizers.l2(1e-4),
                        # activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dense(512, kernel_initializer='normal', activation='relu',
                        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        # bias_regularizer=regularizers.l2(1e-4),
                        # activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(Dense(512, kernel_initializer='normal', activation='relu',
                        # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        # bias_regularizer=regularizers.l2(1e-4),
                        # activity_regularizer=regularizers.l2(1e-5)
                        ))
        # Uncertainty
        model.add(Dense(2,
                        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                        bias_regularizer=regularizers.l2(1e-4),
                        activity_regularizer=regularizers.l2(1e-5)
                        ))
        model.add(tfp.layers.DistributionLambda(
            lambda t: tfd.Normal(loc=t[..., :1],
                                 scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:])),
            name='redshift',
        ))
        negloglik = lambda y, p_y: -p_y.log_prob(y)
        loss = {'redshift': negloglik}

        opt = Adam(lr=self.lr)
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
                validation_data_arr, self.params_exp, additional_metrics_dict=self.metrics_dict, model_wrapper=self,
                tensorboard_callback=self.tensorboard_callback
            )
            self.callbacks = [validation_callback] + self.callbacks
            validation_data = (validation_data_arr[0][0], validation_data_arr[0][1])
        else:
            validation_data = None

        self.network = self.create_network(self.params_exp)
        self.network.fit(X, y, validation_data=validation_data, epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=1)

        # Restore best weights if inference
        if self.params_exp['is_inference']:
            self.network.load_weights(self.model_path)

    def predict(self, X, encoder=None, batch_size=None, scale_data=True):
        batch_size = self.batch_size if batch_size is None else batch_size
        X_to_pred = self.scaler.transform(X) if scale_data else X
        predictions_df = pd.DataFrame()

        indices = [(i + 1) * batch_size for i in range(int(X_to_pred.shape[0] / batch_size))]
        batches = np.split(X_to_pred, indices, axis=0)
        preds_arr = [self.network(batch) for batch in batches]

        # Uncertainty
        predictions_df['Z_PHOTO'] = np.concatenate([preds.mean() for preds in preds_arr])[:, 0]
        predictions_df['Z_PHOTO_STDDEV'] = np.concatenate([preds.stddev() for preds in preds_arr])[:, 0]

        return predictions_df


class AdditionalValidationSets(Callback):
    def __init__(self, validation_sets, cfg, tracked_metric_names=None, additional_metrics_dict=None,
                 model_wrapper=None, batch_size=None, tensorboard_callback=None, verbose=0):
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
        self.cfg = cfg
        self.epoch = []
        self.history = {}
        self.verbose = verbose
        self.batch_size = batch_size
        self.tracked_metric_names = tracked_metric_names
        self.additional_metrics_dict = additional_metrics_dict
        self.model_wrapper = model_wrapper
        self.tensorboard_callback = tensorboard_callback

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

            # Standard model evaluation on tracked metrics
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

            # Make predictions
            predictions = self.model_wrapper.predict(validation_data, scale_data=False)

            # Log scatter plot or confusion matrix
            if self.tensorboard_callback:
                if 'redshift' in validation_targets:
                    predictions['Z'] = validation_targets['redshift']
                    self.log_redshift_scatter(epoch, predictions, validation_set_name)
                # TODO: log only top test confusion matrix
                # if 'category' in validation_targets:
                #     self.log_confusion_matrix(epoch, predictions, validation_targets['category'], validation_set_name)

            # Additional metrics with custom predict function from model wrapper, only present in case of redshift
            if self.additional_metrics_dict:
                for metric_name, metric_func in self.additional_metrics_dict.items():
                    result = metric_func(validation_targets['redshift'], predictions['Z_PHOTO'])
                    value_name = 'val_{}_{}'.format(validation_set_name, metric_name)
                    self.history.setdefault(value_name, []).append(result)
                    logs[value_name] = result

        super().on_epoch_end(epoch, logs)

    def log_redshift_scatter(self, epoch, predictions, test_name):
        log_dir = self.tensorboard_callback.log_dir + '/images'
        # TODO: create just once or close?
        file_writer = tf.summary.create_file_writer(log_dir)
        scatter_plot = redshift_scatter_plot(predictions, z_pred_col='Z_PHOTO', z_pred_stddev_col='Z_PHOTO_STDDEV',
                                             z_max=4, return_figure=True)
        scatter_image = plot_to_image(scatter_plot)
        with file_writer.as_default():
            tf.summary.image('redshift scatter - {}'.format(test_name), scatter_image, step=epoch)

    def log_confusion_matrix(self, epoch, predictions, y_true, test_name):
        log_dir = self.tensorboard_callback.log_dir + '/images'
        file_writer = tf.summary.create_file_writer(log_dir)
        class_names = ['GALAXY', 'QSO', 'STAR']
        y_true_decoded = [class_names[i] for i in np.argmax(y_true, axis=1)]
        cm = confusion_matrix(y_true_decoded, predictions['CLASS_PHOTO'])
        cm_fig = plot_confusion_matrix(cm, classes=class_names, normalize=False, title=None, return_figure=True)
        cm_image = plot_to_image(cm_fig)
        with file_writer.as_default():
            tf.summary.image('confusion matrix - {}'.format(test_name), cm_image, step=epoch)


class CustomTensorBoard(TensorBoard):
    def __init__(self, log_folder, params, is_inference):
        self.log_folder = log_folder
        subfolder = 'inf' if is_inference else 'exp'
        self.log_dir = os.path.join(PROJECT_PATH, 'outputs/tensorboard/{}/{}'.format(subfolder, log_folder))
        super().__init__(log_dir=self.log_dir, profile_batch=0)

        self.params_exp = params
        # self.main_test_name = 'top' if (
        #         params['test_method'] == 'magnitude' and not params['is_inference']) else 'random'

    def on_epoch_end(self, epoch, logs=None):
        logs_to_send = logs.copy()

        # TODO: get adaptive learning rate
        # Add learning rate
        logs_to_send.update({'learning_rate': K.eval(self.model.optimizer.lr)})

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


def decode_clf_preds(y_pred_proba, encoder=None):
    predictions_df = pd.DataFrame()
    class_names = encoder.classes_ if encoder else ['GALAXY', 'QSO', 'STAR']
    for i, c in enumerate(class_names):
        predictions_df['{}_PHOTO'.format(c)] = y_pred_proba[:, i]
    idx_max = np.argmax(y_pred_proba, axis=1)
    predictions_df['CLASS_PHOTO'] = encoder.inverse_transform(idx_max) if encoder else [class_names[i] for i in idx_max]
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
