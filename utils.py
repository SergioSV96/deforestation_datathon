from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Create a function to read the image and return the image and the label
def read_image(image_path, label):
    # Read the image
    img = tf.io.read_file(image_path)
    # Decode the image
    img = tf.image.decode_png(img, channels=3)
    # Convert the image to float32 
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    # MobileNetV2 expects the input to be 224x224
    img = tf.image.resize(img, [224, 224])
    # Normalize the image
    tf.keras.applications.mobilenet_v2.preprocess_input(img)

    return img, label

# Create a function to prepare the dataset
def prepare_dataset(df, batch_size=32):
    # Get the classes from the dataframe
    classes = df['label'].unique()
    # Create a dataset from the dataframe
    dataset = tf.data.Dataset.from_tensor_slices((df['example_path'].values, df['label'].values))
   # Map the read_image function to the dataset
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # One-hot encode the labels
    dataset = dataset.map(lambda image, label: (image, tf.one_hot(label, len(classes))), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(df), seed=42)
    # Cache the dataset
    dataset = dataset.cache()
    # Create batches
    dataset = dataset.batch(batch_size)
    # Prefetch the dataset
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset