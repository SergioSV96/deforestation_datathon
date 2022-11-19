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
    
    # MobileNetV2 settings
    # img = tf.image.resize(img, [224, 224])
    # tf.keras.applications.mobilenet_v2.preprocess_input(img)

    # ResNet50 settings
    # Normalize the image
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img, label

# Create a function to prepare the dataset
def prepare_dataset(df, batch_size=32, augment=False):
    # Get the classes from the dataframe
    classes = df['label'].unique()
    # Create a dataset from the dataframe
    dataset = tf.data.Dataset.from_tensor_slices((df['example_path'].values, df['label'].values))
   # Map the read_image function to the dataset
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # One-hot encode the labels
    dataset = dataset.map(lambda image, label: (image, tf.one_hot(label, len(classes))), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Use data augmentation only on the training set
    if augment:
        # Add data augmentation to the original dataset
        augmentation_techniques = [tf.image.random_flip_left_right, tf.image.random_flip_up_down, tf.image.rot90]
        augmented_datasets = []
        for technique in augmentation_techniques:
            augmented_dataset = dataset.map(lambda image, label: (technique(image), label), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            augmented_datasets.append(augmented_dataset)
        # Concatenate the augmented datasets
        for augmented_dataset in augmented_datasets:
            dataset = dataset.concatenate(augmented_dataset)
    else:
        augmentation_techniques = []
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(df))
    # Cache the dataset
    dataset = dataset.cache()
    # Create batches
    dataset = dataset.batch(batch_size)
    # Prefetch the dataset
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset