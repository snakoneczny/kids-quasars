import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.base import BaseEstimator
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping


def build_rf_clf(params):
    return RandomForestClassifier(
        n_estimators=400, criterion='gini', random_state=491237, n_jobs=12,
    )


def build_rf_reg(params):
    return RandomForestRegressor(
        n_estimators=400, criterion='mse', random_state=491237, n_jobs=12,
    )


def build_xgb_clf(params):
    return XGBClassifier(
        max_depth=7, learning_rate=0.1, gamma=0, min_child_weight=1, colsample_bytree=0.7, subsample=0.8,
        scale_pos_weight=2, reg_alpha=0, reg_lambda=1, n_estimators=100000, objective='multi:softmax', booster='gbtree',
        max_delta_step=0, colsample_bylevel=1, base_score=0.5, random_state=18235, missing=None, verbosity=0, n_jobs=12,
    )


def build_xgb_reg(params):
    # TODO: reg:linear is now deprecated in favor of reg:squarederror
    # All classes
    if not params['specialization']:
        return XGBRegressor(
            max_depth=5, learning_rate=0.1, gamma=0, min_child_weight=20, colsample_bytree=0.5, subsample=1,
            scale_pos_weight=1, reg_alpha=1, reg_lambda=1, n_estimators=100000, objective='reg:linear',
            booster='gbtree', max_delta_step=0, colsample_bylevel=1, colsample_bynode=1, base_score=0.5,
            random_state=1587, missing=None, importance_type='gain', verbosity=0, n_jobs=12,
        )

    elif params['specialization'] == 'QSO':
        # TODO: subsample danger, new random_state?
        return XGBRegressor(
            max_depth=5, learning_rate=0.1, gamma=0, min_child_weight=10, colsample_bytree=0.6, subsample=0.4,
            scale_pos_weight=1, reg_alpha=0, reg_lambda=1, n_estimators=100000, objective='reg:linear',
            booster='gbtree', max_delta_step=0, colsample_bylevel=1, colsample_bynode=1, base_score=0.5,
            random_state=1587, missing=None, importance_type='gain', verbosity=0, n_jobs=12,
        )

    elif params['specialization'] == 'GALAXY':
        return XGBRegressor(
            max_depth=7, learning_rate=0.1, gamma=0, min_child_weight=20, colsample_bytree=1, subsample=1,
            scale_pos_weight=1, reg_alpha=0, reg_lambda=2, n_estimators=100000, objective='reg:linear',
            booster='gbtree', max_delta_step=0, colsample_bylevel=1, colsample_bynode=1, base_score=0.5,
            random_state=1587, missing=None, importance_type='gain', verbosity=0, n_jobs=12,
        )

    else:
        raise Exception('Not implemented redshift specialization: {}'.format(params['specialization']))


class AnnClf(BaseEstimator):

    def __init__(self, params):
        self.params_exp = params
        self.network = None
        self.scaler = MinMaxScaler()
        self.epochs = 2000
        self.patience = 400
        self.batch_size = 256
        self.lr = 0.0001

        log_name = 'class, lr={}, bs={}, {}'.format(self.lr, self.batch_size,
                                                    params['timestamp_start'].replace('_', ' '))
        if params['tag']:
            log_name = '{}, {}'.format(params['tag'], log_name)

        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        tensorboard = CustomTensorBoard(log_folder=log_name, params=self.params_exp)
        self.callbacks = [early_stopping, tensorboard]

    def __create_network(self, params):
        model = Sequential()
        model.add(Dense(60, input_dim=params['n_features'], activation='relu'))
        model.add(Dense(60, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(3, activation='softmax', name='category'))

        opt = Adam(lr=self.lr)

        loss = {'category': 'categorical_crossentropy'}
        model.compile(loss=loss, optimizer=opt, metrics=['categorical_crossentropy', 'accuracy'])

        return model

    def fit(self, X, y, validation_data=None):
        X = self.scaler.fit_transform(X)
        y['category'] = np_utils.to_categorical(y['category'])
        validation_data = (self.scaler.transform(validation_data[0]),
                           {'category': np_utils.to_categorical(validation_data[1]['category'])})

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
        self.epochs = 4000
        self.patience = 400
        self.batch_size = 256
        self.lr = 0.0001

        log_name = 'redshift, lr={}, bs={}, {}'.format(self.lr, self.batch_size,
                                                       params['timestamp_start'].replace('_', ' '))
        if params['tag']:
            log_name = '{}, {}'.format(params['tag'], log_name)

        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)
        tensorboard = CustomTensorBoard(log_folder=log_name, params=self.params_exp)
        self.callbacks = [early_stopping, tensorboard]

    def __create_network(self, params):
        model = Sequential()
        model.add(Dense(80, input_dim=params['n_features'], activation='relu'))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, name='redshift'))

        opt = Adam(lr=self.lr)

        loss = {'redshift': 'mean_squared_error'}
        model.compile(loss=loss, optimizer=opt, metrics=['mean_absolute_error', 'mean_squared_error'])

        return model

    def fit(self, X, y, validation_data=None):
        X = self.scaler.fit_transform(X)
        validation_data = (self.scaler.transform(validation_data[0]),
                           {'redshift': validation_data[1]['redshift']})

        self.network = self.__create_network(self.params_exp)
        self.network.fit(X, y, validation_data=validation_data, epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=1)

    def predict(self, X, encoder=None):
        X = self.scaler.transform(X)
        predictions_df = pd.DataFrame()
        predictions_df['Z_PHOTO'] = self.network.predict(X)[:, 0]
        return predictions_df


class AstroNet(BaseEstimator):

    def __init__(self, params):
        self.params_exp = params
        self.network = None
        self.scaler = MinMaxScaler()
        self.epochs = 4000
        self.patience = 600
        self.batch_size = 256
        self.lr = 0.0001

        log_name = 'lr={}, bs={}, {}'.format(self.lr, self.batch_size, params['timestamp_start'].replace('_', ' '))
        if params['tag']:
            log_name = '{}, {}'.format(params['tag'], log_name)

        tensorboard = CustomTensorBoard(log_folder=log_name, params=self.params_exp)
        early_stopping = EarlyStopping(monitor='val_redshift_loss', patience=self.patience, restore_best_weights=True)
        self.callbacks = [tensorboard, early_stopping]

    def __create_network(self, params):
        inputs = Input(shape=(params['n_features'],))

        # Main branch
        x = Dense(60, activation='relu')(inputs)
        x = Dense(60, activation='relu')(x)
        x = Dense(40, activation='relu')(x)
        x = Dense(40, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        x = Dense(20, activation='relu')(x)

        # Outputs
        y = Dense(10, activation='relu')(x)
        y = Dense(10, activation='relu')(y)
        preds_y = Dense(3, activation='softmax', name='category')(y)

        z = Dense(20, activation='relu')(x)
        z = Dense(20, activation='relu')(z)
        preds_z = Dense(1, name='redshift')(z)

        model = Model(inputs=inputs, outputs=[preds_y, preds_z], name='astronet')

        losses = {
            'category': 'categorical_crossentropy',
            'redshift': 'mean_squared_error',
        }
        loss_weights = {'category': 1.0, 'redshift': 1.0}

        opt = Adam(self.lr)
        model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights,
                      metrics=['categorical_crossentropy', 'accuracy', 'mean_squared_error', 'mean_absolute_error'])

        return model

    def fit(self, X, y, validation_data=None):
        X = self.scaler.fit_transform(X)
        y, validation_data = create_multioutput_data(y, validation_data, self.scaler)
        self.network = self.__create_network(self.params_exp)
        self.network.fit(X, y, validation_data=validation_data, epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=1)

    def predict(self, X, encoder):
        X = self.scaler.transform(X)
        predictions = self.network.predict(X)

        y_pred_proba = predictions[0]
        z_pred = predictions[1]

        predictions_df = decode_clf_preds(y_pred_proba, encoder)
        predictions_df['Z_PHOTO'] = z_pred
        return predictions_df


class CustomTensorBoard(TensorBoard):
    def __init__(self, log_folder, params):
        self.params_exp = params
        log_dir = './tensorboard/exp/{}'.format(log_folder)
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs_to_send = logs.copy()
        # Add learning rate
        logs_to_send.update({'learning rate': K.eval(self.model.optimizer.lr)})

        # Remove artificial logs
        to_pop = ['loss']
        if self.params_exp['pred_class'] and self.params_exp['pred_z']:
            to_pop += [
                # Losses are not needed as we track all metrics with their proper names
                'category_loss',
                'redshift_loss',
                # Below metrics are artificially created due to metrics tracking
                'redshift_categorical_crossentropy',
                'redshift_acc',
                # The same for category
                'category_mean_squared_error',
                'category_mean_absolute_error',
            ]
        for str in to_pop:
            logs_to_send.pop(str, None)
            logs_to_send.pop('val_{}'.format(str), None)

        # Standalone problems, add category or redshift to metric names
        if not(self.params_exp['pred_class'] and self.params_exp['pred_z']):
            t = 'category' if self.params_exp['pred_class'] else 'redshift'
            metrics = ['categorical_crossentropy', 'acc'] if self.params_exp['pred_class'] \
                else ['mean_squared_error', 'mean_absolute_error']
            for metric in metrics:
                logs_to_send['{}_{}'.format(t, metric)] = logs_to_send.pop('{}'.format(metric))
                logs_to_send['val_{}_{}'.format(t, metric)] = logs_to_send.pop('val_{}'.format(metric))

        super().on_epoch_end(epoch, logs_to_send)


# TODO: this function transforms y to categorical and scales X, could it be merged with build_validation_data?
def create_multioutput_data(y, validation_data, scaler):
    y_categorical = np_utils.to_categorical(y['category'])
    y['category'] = y_categorical

    y_val_categorical = np_utils.to_categorical(validation_data[1]['category'])
    validation_data = (scaler.transform(validation_data[0]),
                       {'category': y_val_categorical, 'redshift': validation_data[1]['redshift']})

    return y, validation_data


# TODO: this should be in a pipeline for non ann models, also encoder usage is multiplied in networks and here
def get_single_problem_predictions(model, X, encoder, cfg):
    params_pred = {'ntree_limit': model.best_ntree_limit} if cfg['model'] == 'xgb' and model.best_ntree_limit else {}
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
        'ann-class-redshift': AstroNet,
    }

    name = cfg['model']
    if cfg['pred_class']: name += '-class'
    if cfg['pred_z']: name += '-redshift'

    model = constructors[name](cfg)
    return model
