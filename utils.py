from cProfile import label
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def show_examples(df, n=4):
    # Show one random image from each class in the dataset
    # n: number of images to show
    # df: dataframe with the image name and the class name
    # The path to the dataset
    path = os.environ.get('DATASET_PATH')
    # The list of classes
    classes = df['label'].unique()
    np.random.seed(42)
    plt.figure(figsize=(20, 10))
    # Create a subplot for each class in a 2x2 grid
    # Looping through the classes
    for label in classes:
        # The path to the class
        class_path = path + label + '/'
        # The list of images in the class
        images = os.listdir(class_path)
        # Selecting a random image
        image = np.random.choice(images)
        # Reading the image
        img = plt.imread(class_path + image)
        # Creating a subplot
        plt.subplot(1, 4, classes.index(label) + 1)
        # Showing the image
        plt.imshow(img)
        # Setting the title of the subplot
        plt.title(label)
    plt.show()

# Create a function to read the image and return the image and the label
def read_image(image_name, label):
    # Read the image
    img = tf.io.read_file('images/' + label + '/' + image_name)
    # Convert the image to float32
    img = tf.image.decode_jpeg(img, channels=3)
    # MobileNetV2 expects the input to be 224x224
    img = tf.image.resize(img, [224, 224])
    # Normalize the image
    img = img / 255.0  # type: ignore

    return img, label

# Create a function to prepare the dataset
def prepare_dataset(df, batch_size=32):
    # Get the classes from the dataframe
    classes = df['label'].unique()
    # Create a dataset from the dataframe
    dataset = tf.data.Dataset.from_tensor_slices((df['image_name'].values, df['label'].values))
   # Map the read_image function to the dataset
    dataset = dataset.map(read_image)
    # Label encode the labels
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(tf.constant(classes), tf.range(len(classes))),
        default_value=-1)

    label_encoder = tf.keras.layers.Lambda(lambda x: table.lookup(x))
    dataset = dataset.map(lambda image, label: (image, label_encoder(label)))

    # One-hot encode the labels
    dataset = dataset.map(lambda image, label: (image, tf.one_hot(label, len(classes))))
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=len(df))
    # Create batches
    dataset = dataset.batch(batch_size)
    # Prefetch the dataset
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset