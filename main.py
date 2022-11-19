# Data Preprocessing
# Split the data into train and test sets
from sklearn.model_selection import train_test_split
from models import model
from utils import prepare_dataset
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import datetime

# Create a dataframe from the csv file
df = pd.read_csv('train.csv')

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Batch size
batch_size = 32

# Create the train and test datasets
train_dataset = prepare_dataset(train_df, batch_size=batch_size)
test_dataset = prepare_dataset(test_df, batch_size=batch_size)

# Create the model
model = model(len(df['label'].unique()), size=224)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='categorical_crossentropy',
    metrics=['accuracy', tfa.metrics.F1Score(num_classes=len(df['label'].unique()), average='macro')])

log_dir = f'logs/date_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=50, restore_best_weights=True, mode='max')

# Train the model
model.fit(train_dataset, epochs=32, validation_data=test_dataset, verbose=2, callbacks=[tensorboard_callback, stop_early])

# Evaluate the model
model.evaluate(test_dataset)

# Save the model
model.save('model.h5')