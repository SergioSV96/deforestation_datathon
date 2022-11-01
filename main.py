# Data Preprocessing
# Split the data into train and test sets
from sklearn.model_selection import train_test_split
from models import model
from utils import prepare_dataset
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

df = pd.read_csv('data.csv')

# Split the data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create the train and test datasets
train_dataset = prepare_dataset(train_df, batch_size=32)
test_dataset = prepare_dataset(test_df, batch_size=32)

# Create the model
model = model(size=224)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
    metrics=['accuracy', tfa.metrics.F1Score(num_classes=4, average='macro')])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset, verbose=2)

# Evaluate the model
model.evaluate(test_dataset)

# Save the model
model.save('model.h5')