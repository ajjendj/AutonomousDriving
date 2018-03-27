"""
model.py
-------------------------------------------------------------------------------------------------------------
Trains a keras model to output a steering angle from an input image recorded using a front-facing car camera. 
-------------------------------------------------------------------------------------------------------------
"""

import tensorflow as tf
import pandas as pd
import numpy as np

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Cropping2D, Dropout, Activation, Lambda, Input, Flatten, Dense

from scipy import misc
from sklearn import model_selection

#------------------------------------------------------------------------------------------------------------

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('num_epochs', 20,'The number of epochs when end-to-end training.')
flags.DEFINE_integer('batch_size', 128, 'The batch size.')
flags.DEFINE_integer('samples_per_epoch', 12800,'The number of samples per epoch.')

#------------------------------------------------------------------------------------------------------------

def generate_samples(df):
    """
    Generator function to yield a batch_size number of images and steering labels.

    Args:    
        df (pandas dataframe): Input dataframe consisting of paths to images and steering labels.
    
    Yields:
        images (numpy array): images from the input dataframe.
        steerings (numpy array): steering values corresponding to the images.

    """
    
    while True:
        images, steerings = [],[]
        rows = df.sample(FLAGS.batch_size).iterrows()

        camera_dict = {'left': 0.2, 'right': - 0.2}
        
        for _, row in rows:
            # Read the images and corresponding labels
            if row['center'] != 'center': # avoid reading header of the dataframe
                image = misc.imread('data/'+row['center'])
                steering = float(row['angle'].strip()) #convert steering value to a float
    
                # Only include ~ 10% of images with steering value 0            
                if steering == 0:
                    if np.random.random() < 0.1:
                        images.append(image)
                        steerings.append(steering)
                
                # For every image with a non-zero steering, augment dataset with flips
                #   as well as with recovery images from the non-center cameras
                else:

                    # Add flipped image
                    image_flipped = np.fliplr(image)
                    steering_flipped = -1 * steering
                    images.append(image_flipped)
                    steerings.append(steering_flipped)

                    # Add recovery image from left and right cameras
                    for recovery_direction in list(camera_dict.keys()):
                    
                        fname = 'data/'+row[recovery_direction]
                        fname = fname.replace(" ","") #remove space from filename
                        image_recovery = misc.imread(fname)
                        steering_recovery = float(row['angle']) + camera_dict[recovery_direction]
                        images.append(image_recovery)
                        steerings.append(steering_recovery)
                    
                        # Add flipped recovery image
                        image_flipped = np.fliplr(image_recovery)
                        steering_flipped = -1 * steering_recovery
                        images.append(image_flipped)
                        steerings.append(steering_flipped)
        
        yield np.array(images), np.array(steerings)

#------------------------------------------------------------------------------------------------------------

def main(_):
 
    # Read the training driving log .csv file into a pandas dataframe
    with open('data/driving_log.csv', 'rb') as f:
        df = pd.read_csv(f, header=None, 
                            names=['center', 'left', 'right', 'angle','throttle', 'break', 'speed'])

    # Separate dataframe into train and validation splits
    df_train, df_validation = model_selection.train_test_split(df, test_size=.2)

    print("There are ", len(df_train), "samples for training.")
    print("There are ", len(df_validation), "samples for validation.")


    # Build model. The model conisists of 3 Convolutional Layers and 3 Fully Connected Layers.
    # Preprocessing, consisting of normalizing the images and cropping some of the top and bottom parts,
    #   is also done using Keras layers

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode="same", activation="relu"))
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation="relu"))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same", activation="relu"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))

    # The model is learning using the Adam optimizer minimizing the mse loss
    model.compile(optimizer="adam", loss="mse")
    
    # The model is fit using the fit_generator function
    history = model.fit_generator(
        generate_samples(df_train),
        samples_per_epoch=FLAGS.samples_per_epoch,
        validation_data=generate_samples(df_validation),
        nb_epoch=FLAGS.num_epochs,
        verbose=1,
        nb_val_samples=df_validation.shape[0])

    # Write the model to disk
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')
    print('Model has been saved.')

#------------------------------------------------------------------------------------------------------------

# Call main function

if __name__ == '__main__':
    tf.app.run()
