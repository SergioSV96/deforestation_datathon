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

import models


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
train_dataset = utils.prepare_dataset(train_df, batch_size=batch_size, augment=True)
test_dataset = utils.prepare_dataset(test_df, batch_size=batch_size)

# Print the size of the train and test datasets
print(f'Train dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}')

NUM_CLASSES = len(df['label'].unique())

# Take the best model from the tuning script and use it to fine-tune the model.

# Create the base model from the pre-trained (from our transfer learning script) model MobileNet V2
# Load model with weights
model = tf.keras.models.load_model('models/transfer_learning/transfer_learning.h5')

# Unfreeze the base model
model.trainable = True

# Train the model with the new layers
# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
    metrics=['accuracy', tfa.metrics.F1Score(num_classes=len(df['label'].unique()), average='macro')])

# Callbacks
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=50, restore_best_weights=True, mode='max')

callbacks = [stop_early]

# Train the model
history = model.fit(train_dataset, epochs=200, class_weight=class_weights, validation_data=test_dataset, verbose=2, callbacks=callbacks)

# Evaluate the model on the validation set.
eval_result = model.evaluate(test_dataset, verbose=0)

# Save the model
model.save(f'models/tuning/fine_tuned_{np.round(eval_result[2], 4)}.h5') # Save the model with the best F1 score

# Predict on the submission set
submission_df = pd.read_csv('test.csv')

submission_dataset = utils.prepare_dataset(submission_df, batch_size=batch_size, augment=False, shuffle=False)

predictions = model.predict(submission_dataset)

# Save the predictions to a json file called predictions.json with the following format:
# {
#   "target: {
#       "0": 0,
#       "1": 2,
#       "2": 1,
#       "3": 0,
#       "4": 1,
#       ...
# }

# Create a dictionary with the predictions
predictions_dict = {}

for i, prediction in enumerate(predictions):
    predictions_dict[i] = np.argmax(prediction)

# Save the predictions to a json file
import json

with open('predictions.json', 'w') as f:
    json.dump(predictions_dict, f)