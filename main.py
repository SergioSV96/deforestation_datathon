# Data Preprocessing
# Split the data into train and test sets
from sklearn.model_selection import train_test_split
import models
import utils
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

from focal_loss import SparseCategoricalFocalLoss, BinaryFocalLoss

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# os.system("!rm -rf logs")

# Create a dataframe from the csv file
df = pd.read_csv('train.csv')

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
class_weights = dict(enumerate(class_weights))


# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])

# Batch size
batch_size = 64

# Create the train and test datasets
train_dataset = utils.prepare_dataset(train_df, batch_size=batch_size)
test_dataset = utils.prepare_dataset(test_df, batch_size=batch_size)

# Create the model
model = models.model(len(df['label'].unique()), size=224)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy',
    metrics=['accuracy', tfa.metrics.F1Score(num_classes=len(df['label'].unique()), average='macro')])

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy',
#     metrics=['accuracy', tfa.metrics.F1Score(num_classes=len(df['label'].unique()), average='macro')])


# Callbacks
log_dir = f'logs/date_{datetime.datetime.now().strftime("%H:%M:%S")}'
file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')  # type: ignore
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=50, restore_best_weights=True, mode='max')
cm_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: tensorboard_utils.log_confusion_matrix(epoch, logs, model, test_dataset, file_writer_cm))

# Train the model
model.fit(train_dataset, epochs=32, class_weight=class_weights, validation_data=test_dataset, verbose=2, callbacks=[tensorboard_callback, stop_early, cm_callback])

# Evaluate the model
model.evaluate(test_dataset)
# print(model.predict(test_dataset))

# Save the model
# model.save('model.h5')