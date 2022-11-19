import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import json

import preprocessing

def create_submission(model_name, batch_size, num_classes):
    model = tf.keras.models.load_model('models/tuning/' + model_name + '.h5')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
        metrics=['accuracy', tfa.metrics.F1Score(num_classes=num_classes, average='macro')])

    # Predict on the submission set
    submission_df = pd.read_csv('test.csv')
    submission_dataset = preprocessing.prepare_test_dataset(submission_df, batch_size=batch_size)

    predictions = model.predict(submission_dataset)

    # Create a dictionary with the predictions
    predictions_dict = {}

    for i, prediction in enumerate(predictions):
        predictions_dict[i] = int(np.argmax(prediction))
    
    predictions_dict = {'target': predictions_dict}

    # Save the predictions to a json file
    with open(f'predictions_{model_name}.json', 'w+') as f:
        json.dump(predictions_dict, f)

create_submission(model_name='fine_tuned_0.766', batch_size=64, num_classes=3)