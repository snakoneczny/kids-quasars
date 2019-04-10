import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils


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
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=18235, missing=None
    )


class AnnClf(BaseEstimator):

    def __init__(self, params):
        self.params = params
        self.scaler = MinMaxScaler()
        self.network = None

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
        model.add(Dense(3, activation='softmax', name='category_output'))

        loss = {'category_output': 'categorical_crossentropy'}

        opt = Adam(lr=0.001)

        model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        return model

    def fit(self, X, y, validation_data=None):
        X = self.scaler.fit_transform(X)
        y['category_output'] = np_utils.to_categorical(y['category_output'])
        self.network = self.__create_network(self.params)

        validation_data = (self.scaler.transform(validation_data[0]),
                           {'category_output': np_utils.to_categorical(validation_data[1]['category_output'])})

        self.network.fit(X, y, validation_data=validation_data, epochs=50, batch_size=32, verbose=1)

    def predict(self, X, encoder):
        X = self.scaler.transform(X)
        y_pred_proba = self.network.predict(X)
        predictions_df = decode_clf_preds(y_pred_proba, encoder)
        return predictions_df


class AstroNet(BaseEstimator):

    def __init__(self, params):
        self.params = params
        self.scaler = MinMaxScaler()
        self.network = None

    def __create_network(self, params):
        inputs = Input(shape=(params['n_features'],))

        # Main branch
        x = Dense(80, activation='relu')(inputs)
        x = Dense(80, activation='relu')(x)
        x = Dense(80, activation='relu')(x)
        x = Dense(60, activation='relu')(x)
        x = Dense(60, activation='relu')(x)

        # Class branch
        y = Dense(40, activation='relu')(x)
        y = Dense(40, activation='relu')(y)

        # Redshift branch
        z = Dense(40, activation='relu')(x)
        z = Dense(40, activation='relu')(z)

        preds_category = Dense(3, activation='softmax', name='category_output')(y)
        preds_redshift = Dense(1, name='redshift_output')(z)

        model = Model(inputs=inputs, outputs=[preds_category, preds_redshift], name='astronet')

        losses = {
            'category_output': 'categorical_crossentropy',
            'redshift_output': 'mean_squared_error',
        }
        loss_weights = {'category_output': 1.0, 'redshift_output': 1.0}

        opt = Adam(lr=0.001)
        model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=['accuracy'])

        return model

    def fit(self, X, y, validation_data=None):
        X = self.scaler.fit_transform(X)
        y['category_output'] = np_utils.to_categorical(y['category_output'])
        self.network = self.__create_network(self.params)

        validation_data = (self.scaler.transform(validation_data[0]),
                           {'category_output': np_utils.to_categorical(validation_data[1]['category_output']),
                            'redshift_output': validation_data[1]['redshift_output']})

        self.network.fit(X, y, validation_data=validation_data, epochs=100, batch_size=32, verbose=1)

    def predict(self, X, encoder):
        X = self.scaler.transform(X)
        predictions = self.network.predict(X)

        y_pred_proba = predictions[0]
        z_pred = predictions[1]

        predictions_df = decode_clf_preds(y_pred_proba, encoder)
        predictions_df['Z_PHOTO'] = z_pred
        return predictions_df


# TODO: this should be in a pipeline for non ann models
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


def get_model_constructor(cfg):
    constructors = {
        'rf-class': build_rf_clf,
        'rf-redshift': build_rf_reg,
        'xgb-class': build_xgb_clf,
        'ann-class': AnnClf,
        'ann-class-redshift': AstroNet,
    }

    name = cfg['model']
    if cfg['pred_class']: name += '-class'
    if cfg['pred_z']: name += '-redshift'

    return constructors[name]
