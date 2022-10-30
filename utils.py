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
    path = 'images/'
    # The list of classes
    classes = ['cloudy', 'desert', 'green_area', 'water']
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
    # This is resized to 224x224 because the MobileNetV2 model takes images of this size as input
    # Resize the image
    img = tf.image.resize(img, [224, 224])
    # Normalize the image
    img = img / 255.0  # type: ignore
    return img, label

# Create a function to prepare the dataset
def prepare_dataset(df):
    # Create a dataset
    dataset = tf.data.Dataset.from_tensor_slices((df['image_name'].values, df['label'].values))
    # Map the dataset
    dataset = dataset.map(read_image)
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=1000)
    # Batch the dataset
    dataset = dataset.batch(32)
    # Prefetch the dataset
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset