# Using tensorflow to create a model to classify satellite images into 4 classes: cloudy, desert, green_area and water
# We are using transfer learning to create our model using the MobileNetV2 model as the base model and adding a custom head to it to classify the images into 4 classes
# We are using the Adam optimizer and the sparse categorical crossentropy loss function
# We are using the F1 score as the metric to evaluate our model
#

import tensorflow as tf

# class Model(tf.keras.Model):
#     def __init__(self, size=224):
#         super(Model, self).__init__()
#         # Create the base model from the pre-trained model MobileNet V2
#         self.base_model = tf.keras.applications.MobileNetV2(
#             input_shape=(size, size, 3), include_top=False, weights='imagenet')

#         # Freeze the base model
#         self.base_model.trainable = False

#         # Create new model on top
#         self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#         self.dense_layer = tf.keras.layers.Dense(4, activation='softmax')

#     def call(self, inputs):
#         x = self.base_model(inputs, training=False)
#         x = self.global_average_layer(x)
#         x = self.dense_layer(x)

#         return x

def model(num_classes, size=224):
    # Create the base model from the pre-trained model MobileNet V2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(size, size, 3), include_top=False, weights='imagenet')

    # Freeze the base model
    base_model.trainable = False

    # Create new model on top
    inputs = tf.keras.Input(shape=(size, size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Add dense layers on top
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)

    return model