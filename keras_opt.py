from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from sklearn.metrics import roc_auc_score
from keras_tqdm import TQDMCallback
from keras.callbacks import EarlyStopping

from hyperopt import Trials, STATUS_OK, tpe
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
import utils

def k_evaluate(model, x, y):
    y_pred = model.predict_classes(x)
    score = roc_auc_score(y_pred, y)
    print("%s: %0.2f - [%s]" % (roc_auc_score.__name__.upper(), score, 'Keras MLP'))
    return score

def create_model(x_train, y_train, x_test, y_test):
    """
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    """
    model = Sequential()
    model.add(Dense(128, activation='sigmoid',input_shape=(80,)))
    # If we choose 'four', add an additional fourth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense({{choice([256, 512, 1024])}}))
        model.add(Activation({{choice(['tanh', 'sigmoid'])}}))
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(x_train, y_train,
              batch_size={{choice([2048, 4096])}},
              epochs=1,
              verbose=2,
              validation_data=(x_test, y_test))
              #callbacks = [early, progress])
    
    score = k_evaluate(model)
    
    return {'loss': -score, 'status': STATUS_OK, 'model': model}

def data():
    features_set = utils.load_features_set()
    y = features_set['SeriousDlqin2yrs'].as_matrix()
    x = features_set.drop(['SeriousDlqin2yrs'], axis=1).as_matrix()
    x_train, x_test, y_train, y_test = utils.split_dataset(x, y)

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':

    best_run, best_model = optim.minimize(model=create_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=5,
                                        trials=Trials())

    print("Evalutation of best performing model:")
    print(k_evaluate(best_model, x_test_f, y_test_f))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)