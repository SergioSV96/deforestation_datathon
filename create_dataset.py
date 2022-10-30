# We are using https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification to train our model and predict the class of the satellite image
# 4 classes are present in the dataset: cloudy, desert, green_area and water
# Let's create a csv file with the image name and the class name for each image in the dataset
# We will use this csv file to train our model

# Importing the required libraries
import os
import pandas as pd

# Download the dataset manually from https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification

df = pd.DataFrame(columns=['image_name', 'label'])

# The path to the dataset
path = 'images/'

# The list of classes
classes = ['cloudy', 'desert', 'green_area', 'water']

# Looping through the classes
for class_name in classes:
    # The path to the class
    class_path = path + class_name + '/'

    # The list of images in the class
    images = os.listdir(class_path)

    # Looping through the images
    for image in images:
        # Adding the image name and the class name to the dataframe
        df = df.append({'image_name': image, 'class_name': class_name}, ignore_index=True)

# Saving the dataframe to a csv file
df.to_csv('data.csv', index=False)

# Let's see the first 5 rows of the dataframe
df.head()
