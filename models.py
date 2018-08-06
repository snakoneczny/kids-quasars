from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator
from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils


def build_rf_clf(params):
    return RandomForestClassifier(
        n_estimators=400, random_state=491237, criterion='entropy', n_jobs=8,
        # class_weight={'QSO': 0.4, 'STAR': 0.4, 'GALAXY': 0.2}
    )


def build_rf_reg(params):
    return RandomForestRegressor(
        n_estimators=100, random_state=491237, n_jobs=8,
    )


def build_xgb_clf(params):
    return XGBClassifier(
        max_depth=7, learning_rate=0.1, n_estimators=200, objective='multi:softmax', booster='gbtree', n_jobs=8,
        gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=18235, missing=None
    )


def build_ann_clf(params):
    model = Sequential()
    model.add(Dense(20, input_dim=params['n_features'], activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_astronet(params):
    return AstroNet(params)


class AstroNet(BaseEstimator):

    def __init__(self, params):
        self.params = params
        self.scaler = MinMaxScaler()
        self.network = None

    def __create_network(self, params):
        inputs = Input(shape=(params['n_features'],))

        x = Dense(80, activation='relu')(inputs)
        x = Dense(80, activation='relu')(x)
        x = Dense(80, activation='relu')(x)
        x = Dense(60, activation='relu')(x)
        x = Dense(40, activation='relu')(x)

        # y = Dense(20, activation='relu')(x)
        # z = Dense(20, activation='relu')(x)

        preds_category = Dense(3, activation='softmax', name='category_output')(x)
        preds_redshift = Dense(1, name='redshift_output')(x)

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

        self.network.fit(X, y, validation_data=validation_data, epochs=200, batch_size=64, verbose=1)

    def predict(self, X):
        X = self.scaler.transform(X)
        return self.network.predict(X)


MODEL_CONSTRUCTORS = {
    'rf-clf': build_rf_clf,
    'rf-reg': build_rf_reg,
    'xgb-clf': build_xgb_clf,
    'ann-clf': build_ann_clf,
    'astronet': build_astronet,
}
