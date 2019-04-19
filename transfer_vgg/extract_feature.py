import sys
sys.path.append(".")

from pathlib import Path
import numpy as np
import joblib
from keras.preprocessing import image
from keras.applications import vgg16
from keras.utils import to_categorical
from random import shuffle

# from PIL import Image
# sys.modules['Image'] = Image

train_images = []
train_labels = []
test_images = []
test_labels = []
label_name_to_index = {}
label_index_to_name = {}

# # Load all the not-dog images
def load_img(path, label):
    count = 0
    images = sorted(path.glob("*.jpg"))
    shuffle(images)
    for img in images:
        # Load the image from disk
        img = image.load_img(img, target_size=(224, 224))

        # Convert the image to a numpy array
        image_array = image.img_to_array(img)

        if count < 80:
            test_images.append(image_array)
            test_labels.append(label)
            count = count + 1
        else:
            # Add the image to the list of images
            train_images.append(image_array)

            # For each 'not dog' image, the expected value should be 0
            train_labels.append(label)

def load_dir():
# Path to folders with training data
    p = Path('../train')
    label_index = 0
    for child in p.iterdir():
        if child.is_dir():
            print(child.name)
            label_name_to_index[child.name] = label_index
            label_index_to_name[label_index] = child.name
            load_img(child, label_index)
            label_index = label_index + 1

def process_data():
    # Create a single numpy array with all the images we loaded
    x_train = np.array(train_images)
    x_test = np.array(test_images)

    # Also convert the labels to a numpy array
    y_train = to_categorical(train_labels)
    y_test= to_categorical(test_labels)

    # y_train = np.array(cat_labels)

    # Normalize image data to 0-to-1 range
    x_train = vgg16.preprocess_input(x_train)
    x_test = vgg16.preprocess_input(x_test)


    # Load a pre-trained neural network to use as a feature extractor
    pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Extract features for each image (all in one pass)
    features_x_train = pretrained_nn.predict(x_train)
    features_x_test = pretrained_nn.predict(x_test)


    # Save the array of extracted features to a file
    joblib.dump(features_x_train, "x_train2.dat")
    joblib.dump(features_x_test, "x_test2.dat")


    # Save the matching array of expected values to a file
    joblib.dump(y_train, "y_train2.dat")
    joblib.dump(y_test, "y_test2.dat")


if __name__ == '__main__':
    load_dir()
    process_data()

