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
        n_estimators=400, random_state=491237, criterion='entropy', n_jobs=12,
        # class_weight={'QSO': 0.4, 'STAR': 0.4, 'GALAXY': 0.2}
    )


def build_rf_reg(params):
    return RandomForestRegressor(
        n_estimators=400, random_state=491237, n_jobs=12,
    )


def build_xgb_clf(params):
    return XGBClassifier(
        max_depth=7, learning_rate=0.1, n_estimators=200, objective='multi:softmax', booster='gbtree', n_jobs=12,
        gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=18235, missing=None, verbosity=1,
    )


def build_xgb_reg(params):
    return XGBRegressor(max_depth=7, learning_rate=0.1, n_estimators=200, verbosity=0, objective='reg:linear',
                        booster='gbtree', n_jobs=12, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
                        colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1,
                        scale_pos_weight=1, base_score=0.5, random_state=1587, missing=None, importance_type='gain')


class AnnClf(BaseEstimator):

    def __init__(self, params):
        self.params = params
        self.network = None
        self.scaler = MinMaxScaler()
        self.epochs = 2000
        self.batch_size = 256
        self.lr = 0.0001

        logs_name = 'class, lr={}, bs={}, {}'.format(self.lr, self.batch_size,
                                                     params['timestamp_start'].replace('_', ' '))
        if params['tag']:
            logs_name = '{}, {}'.format(params['tag'], logs_name)

        tensorboard = CustomTensorBoard(log_dir='./log_ann/{}'.format(logs_name))
        early_stopping = EarlyStopping(monitor='val_loss', patience=400, restore_best_weights=True)
        self.callbacks = [tensorboard, early_stopping]

    def __create_network(self, params):
        model = Sequential()
        model.add(Dense(40, input_dim=params['n_features'], activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(3, activation='softmax', name='category'))

        opt = Adam(lr=self.lr)

        loss = {'category': 'categorical_crossentropy'}
        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        return model

    def fit(self, X, y, validation_data=None):
        X = self.scaler.fit_transform(X)
        y['category'] = np_utils.to_categorical(y['category'])
        validation_data = (self.scaler.transform(validation_data[0]),
                           {'category': np_utils.to_categorical(validation_data[1]['category'])})

        self.network = self.__create_network(self.params)
        self.network.fit(X, y, validation_data=validation_data, epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=1)

    def predict(self, X, encoder):
        X = self.scaler.transform(X)
        y_pred_proba = self.network.predict(X)
        predictions_df = decode_clf_preds(y_pred_proba, encoder)
        return predictions_df


class AnnReg(BaseEstimator):

    def __init__(self, params):
        self.params = params
        self.network = None
        self.scaler = MinMaxScaler()
        self.epochs = 4000
        self.batch_size = 256
        self.lr = 0.0001

        logs_name = 'redshift, lr={}, bs={}, {}'.format(self.lr, self.batch_size,
                                                     params['timestamp_start'].replace('_', ' '))
        if params['tag']:
            logs_name = '{}, {}'.format(params['tag'], logs_name)

        tensorboard = CustomTensorBoard(log_dir='./log_ann/{}'.format(logs_name))
        early_stopping = EarlyStopping(monitor='val_loss', patience=600, restore_best_weights=True)
        self.callbacks = [tensorboard, early_stopping]

    def __create_network(self, params):
        model = Sequential()
        model.add(Dense(80, input_dim=params['n_features'], activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(20, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(1, name='redshift'))

        opt = Adam(lr=self.lr)

        loss = {'redshift': 'mean_squared_error'}
        model.compile(loss=loss, optimizer=opt)

        return model

    def fit(self, X, y, validation_data=None):
        X = self.scaler.fit_transform(X)
        validation_data = (self.scaler.transform(validation_data[0]),
                           {'redshift': validation_data[1]['redshift']})

        self.network = self.__create_network(self.params)
        self.network.fit(X, y, validation_data=validation_data, epochs=self.epochs, batch_size=self.batch_size,
                         callbacks=self.callbacks, verbose=1)

    def predict(self, X, encoder=None):
        X = self.scaler.transform(X)
        predictions_df = pd.DataFrame()
        predictions_df['Z_PHOTO'] = self.network.predict(X)[:, 0]
        return predictions_df


class AstroNet(BaseEstimator):

    def __init__(self, params):
        self.params = params
        self.network = None
        self.scaler = MinMaxScaler()
        self.epochs = 1000
        self.batch_size = 256
        self.lr = 0.0001

        logs_name = 'lr={}, bs={}, {}'.format(self.lr, self.batch_size, params['timestamp_start'].replace('_', ' '))
        if params['tag']:
            logs_name = '{}, {}'.format(params['tag'], logs_name)

        tensorboard = AstronetTensorBoard(log_dir='./log_ann/{}'.format(logs_name))
        early_stopping = EarlyStopping(monitor='val_category_loss', patience=400, restore_best_weights=True)
        self.callbacks = [tensorboard, early_stopping]

    def __create_network(self, params):
        inputs = Input(shape=(params['n_features'],))

        # Main branch
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)

        # First outputs
        y = Dense(16, activation='relu')(x)
        y = Dense(8, activation='relu')(y)
        preds_y_1 = Dense(3, activation='softmax', name='category_1')(y)

        z = Dense(32, activation='relu')(x)
        z = Dense(16, activation='relu')(z)
        preds_z_1 = Dense(1, name='redshift_1')(z)

        # Continue main branch
        x = Dense(32, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        x = Dense(32, activation='relu')(x)

        # Second outputs
        y = Dense(16, activation='relu')(x)
        y = Dense(8, activation='relu')(y)
        preds_y_2 = Dense(3, activation='softmax', name='category_2')(y)

        z = Dense(32, activation='relu')(x)
        z = Dense(16, activation='relu')(z)
        preds_z_2 = Dense(1, name='redshift_2')(z)

        # preds_category = Dense(3, activation='softmax', name='category')(y)
        # preds_redshift = Dense(1, name='redshift')(z)

        model = Model(inputs=inputs, outputs=[preds_y_1, preds_y_2, preds_z_1, preds_z_2], name='astronet')

        losses = {
            'category_1': 'categorical_crossentropy',
            'category_2': 'categorical_crossentropy',
            'redshift_1': 'mean_squared_error',
            'redshift_2': 'mean_squared_error',
        }
        loss_weights = {'category_1': 1.0, 'category_2': 1.0, 'redshift_1': 1.0, 'redshift_2': 1.0}

        opt = Adam(self.lr)
        model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

        return model

    def fit(self, X, y, validation_data=None):
        X = self.scaler.fit_transform(X)
        y, validation_data = create_multioutput_data(y, validation_data, self.scaler)
        self.network = self.__create_network(self.params)
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


# TODO: pass prediction type, and fix logs
class CustomTensorBoard(TensorBoard):
    def __init__(self, log_dir):
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Add learning rate
        logs.update({'learning rate': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class AstronetTensorBoard(CustomTensorBoard):
    def __init__(self, log_dir):
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Remove not useful logs
        logs.pop('loss', None)  # There's already category loss
        logs.pop('val_loss', None)  # There's already category loss
        logs.pop('redshift_acc', None)
        logs.pop('val_redshift_acc', None)
        super().on_epoch_end(epoch, logs)


# TODO: this function transforms y to categorical and scales X, could it be merged with build_validation_data?
def create_multioutput_data(y, validation_data, scaler):
    y_categorical = np_utils.to_categorical(y['category'])
    y['category_1'] = y_categorical
    y['category_2'] = y_categorical
    y['category'] = y_categorical
    y['redshift_1'] = y['redshift']
    y['redshift_2'] = y['redshift']

    y_val_categorical = np_utils.to_categorical(validation_data[1]['category'])
    validation_data = (scaler.transform(validation_data[0]),
                       {'category': y_val_categorical,
                        'category_1': y_val_categorical,
                        'category_2': y_val_categorical,
                        'redshift': validation_data[1]['redshift'],
                        'redshift_1': validation_data[1]['redshift'],
                        'redshift_2': validation_data[1]['redshift']
                        })

    return y, validation_data


# TODO: this should be in a pipeline for non ann models, also encoder usage is multiplied in networks and here
def get_single_problem_predictions(model, X, encoder, cfg):
    if cfg['pred_class']:
        y_pred_proba = model.predict_proba(X)
        predictions = decode_clf_preds(y_pred_proba, encoder)
    else:
        predictions = pd.DataFrame()
        predictions['Z_PHOTO'] = model.predict(X)
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
