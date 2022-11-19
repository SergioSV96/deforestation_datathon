from sklearn.model_selection import train_test_split
import models
import deforestation_datathon.preprocessing as preprocessing
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import datetime
from sklearn.utils import class_weight
import tensorboard_utils

import random
import numpy as np
import tensorflow as tf

import models


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def train(model, batch_size, learning_rate=0.0001, save=True):
    # Load training data
    df = pd.read_csv('train.csv')

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
    class_weights = dict(enumerate(class_weights))

    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])

    # Create the train and test datasets
    train_dataset = preprocessing.prepare_dataset(train_df, batch_size=batch_size, augment=True)
    test_dataset = preprocessing.prepare_dataset(test_df, batch_size=batch_size)

    # Compile the model
    model.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
        metrics=['accuracy', tfa.metrics.F1Score(num_classes=len(df['label'].unique()), average='macro')])

    # Callbacks
    log_dir = f'logs/fine_tuning/MobileNetV2_pruning_v3_17_0.6932_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')  # type: ignore

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: tensorboard_utils.log_confusion_matrix(epoch, logs, model, test_dataset, file_writer_cm))
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=50, restore_best_weights=True, mode='max')
    callbacks = [tensorboard_callback, stop_early, cm_callback]

    # Train the model
    history = model.fit(train_dataset, epochs=200, class_weight=class_weights, validation_data=test_dataset, verbose=2, callbacks=callbacks)

    # Evaluate the model on the validation set.
    eval_result = model.evaluate(test_dataset, verbose=0)

    if save:
        # Save the model
        model.save(f'models/tuning/fine_tuned_{np.round(eval_result[2], 4)}.h5') # Save the model with the best F1 score
    
    return model, eval_result


# Fine-tuning the best model

model = tf.keras.models.load_model('models/tuning/MobileNetV2_pruning_v3_17_0.6932.h5')
train(model, batch_size=64, learning_rate=0.0001)