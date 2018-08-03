from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier

from keras.models import Sequential, Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD


def build_rf_clf(params):
    return RandomForestClassifier(
        n_estimators=400, random_state=491237, criterion='entropy', n_jobs=8,
        # class_weight={'QSO': 0.4, 'STAR': 0.4, 'GALAXY': 0.2}
    )


def build_rf_reg(params):
    return RandomForestRegressor(
        n_estimators=400, random_state=491237, criterion='entropy', n_jobs=8,
    )


def build_xgb(params):
    return XGBClassifier(
        max_depth=7, learning_rate=0.1, n_estimators=200, objective='multi:softmax', booster='gbtree', n_jobs=8,
        gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
        reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=18235, missing=None
    )


def build_ann(params):
    model = Sequential()
    model.add(Dense(20, input_dim=params['n_features'], activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def build_astronet(params):
    inputs = Input(shape=(params['n_features'],))

    x = Dense(20, activation='relu')(inputs)
    x = Dense(20, activation='relu')(x)

    y = Dense(10, activation='relu')(x)
    z = Dense(10, activation='relu')(x)

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


MODEL_CONSTRUCTORS = {
    'rf': build_rf_reg,
    'xgb': build_xgb,
    'ann': build_ann,
    'astronet': build_astronet,
}
