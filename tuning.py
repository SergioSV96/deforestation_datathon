import optuna
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
import random

import utils

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BATCH_SIZE = 32

NUM_CLASSES = 3
SIZE = 224

EPOCHS = 100


def get_data():
    df = pd.read_csv('train.csv')

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])

    train_dataset = utils.prepare_dataset(train_df, batch_size=BATCH_SIZE, augment=True)
    test_dataset = utils.prepare_dataset(test_df, batch_size=BATCH_SIZE)

    return train_dataset, test_dataset, df

def create_model(trial):
    # We optimize the numbers of layers and their units.
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(SIZE, SIZE, 3), include_top=False, weights='imagenet')

    # Freeze the base model
    base_model.trainable = False

    # Create new model on top
    inputs = tf.keras.Input(shape=(SIZE, SIZE, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    n_layers = trial.suggest_int("n_layers", 1, 3)
    # Add dense layers on top
    for i in range(n_layers):
        n_units = trial.suggest_int("n_units_l{}".format(i), 256, 1024)
        x = tf.keras.layers.Dense(n_units, activation='relu')(x)
        dropout_prob = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
        x = tf.keras.layers.Dropout(dropout_prob)(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)


def create_optimizer(trial):
    # We optimize the choice of optimizers as well as their parameters.
    kwargs = {}
    optimizer_options = ["RMSprop", "Adam", "SGD"]
    optimizer_selected = trial.suggest_categorical("optimizer", optimizer_options)
    if optimizer_selected == "RMSprop":
        kwargs["learning_rate"] = trial.suggest_float(
            "rmsprop_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["decay"] = trial.suggest_float("rmsprop_decay", 0.85, 0.99)
        kwargs["momentum"] = trial.suggest_float("rmsprop_momentum", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "Adam":
        kwargs["learning_rate"] = trial.suggest_float("adam_learning_rate", 1e-5, 1e-1, log=True)
    elif optimizer_selected == "SGD":
        kwargs["learning_rate"] = trial.suggest_float(
            "sgd_opt_learning_rate", 1e-5, 1e-1, log=True
        )
        kwargs["momentum"] = trial.suggest_float("sgd_opt_momentum", 1e-5, 1e-1, log=True)

    optimizer = getattr(tf.optimizers, optimizer_selected)(**kwargs)
    return optimizer


def objective(trial):
    train_dataset, test_dataset, df = get_data()

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
    class_weights = dict(enumerate(class_weights))

    f1_score_macro = tfa.metrics.F1Score(num_classes=len(df['label'].unique()), average='macro')

    model = create_model(trial)
    optimizer = create_optimizer(trial)

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy", f1_score_macro])

    # Early stopping
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=7, restore_best_weights=True, mode='max')
    # Model fit with checkpoint callback
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=test_dataset, class_weight=class_weights, callbacks=[stop_early], verbose=2)

    history.history['val_f1_score']

    # Evaluate the model on the validation set.
    eval_result = model.evaluate(test_dataset, verbose=0)

    return eval_result[2]  # F1 score macro


if __name__ == "__main__":
    storage_name = "sqlite:///example.db"
    study_name = 'MobileNetV2'

    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
    study.optimize(objective, n_trials=25, timeout=600)

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))