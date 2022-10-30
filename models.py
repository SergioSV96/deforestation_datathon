# Using tensorflow to create a model to classify satellite images into 4 classes: cloudy, desert, green_area and water
# We are using transfer learning to create our model using the MobileNetV2 model as the base model and adding a custom head to it to classify the images into 4 classes
# We are using the Adam optimizer and the sparse categorical crossentropy loss function
# We are using the F1 score as the metric to evaluate our model
#

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

def model(size=224):
    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights='imagenet')

    # Freeze the base model
    base_model.trainable = False

    # Create new model on top
    inputs = tf.keras.Input(shape=(size, size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    return model