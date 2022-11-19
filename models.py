import tensorflow as tf


class MobileNetV2(tf.keras.Model):
    def __init__(self, num_classes, size=224):
        super(MobileNetV2, self).__init__()
        # Create the base model from the pre-trained model MobileNet V2
        self.base_model = tf.keras.applications.MobileNetV2(
            input_shape=(size, size, 3), include_top=False, weights='imagenet')

        # Freeze the base model
        self.base_model.trainable = False

        # Create new model on top
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

        self.dense_1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.dense_2 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(0.2)
        self.dense_3 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_3 = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_average_layer(x)

        # Add dense layers on top
        x = self.dense_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        x = self.dense_3(x)
        x = self.dropout_3(x)

        x = self.output_layer(x)

        return x

class ResNet50(tf.keras.Model):
    def __init__(self, num_classes, size=224):
        super(ResNet50, self).__init__()
        # Create the base model from the pre-trained model MobileNet V2
        self.base_model = tf.keras.applications.ResNet50(
            input_shape=(size, size, 3), include_top=False, weights='imagenet')

        # Freeze the base model
        self.base_model.trainable = False

        # Create new model on top
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

        self.dense_1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.dense_2 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(0.2)
        self.dense_3 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_3 = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        x = self.base_model(inputs, training=False)
        x = self.global_average_layer(x)

        # Add dense layers on top
        x = self.dense_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        x = self.dense_3(x)
        x = self.dropout_3(x)

        x = self.output_layer(x)

        return x

class EfficientNetB0(tf.keras.Model):
    def __init__(self, num_classes, size=224):
        super(EfficientNetB0, self).__init__()
        # Create the base model from the pre-trained model MobileNet V2
        self.base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(size, size, 3), include_top=False, weights='imagenet')

        # Freeze the base model
        self.base_model.trainable = False

        # Create new model on top
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

        self.dense_1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dropout_1 = tf.keras.layers.Dropout(0.2)
        self.dense_2 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout_2 = tf.keras.layers.Dropout(0.2)
        self.dense_3 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout_3 = tf.keras.layers.Dropout(0.2)

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.global_average_layer(x)

        # Add dense layers on top
        x = self.dense_1(x)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        x = self.dense_3(x)
        x = self.dropout_3(x)

        x = self.output_layer(x)

        return x
