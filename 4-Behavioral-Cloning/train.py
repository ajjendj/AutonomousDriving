import tensorflow as tf
import pandas as pd
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Cropping2D, Dropout, Activation, Lambda
from keras.layers import Input, Flatten, Dense, ELU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

from scipy import misc
from skimage import color

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('features_epochs', 1,
                     'The number of epochs when training features.')
flags.DEFINE_integer('full_epochs', 50,
                     'The number of epochs when end-to-end training.')
flags.DEFINE_integer('batch_size', 128, 'The batch size.')
flags.DEFINE_integer('samples_per_epoch', 12800,
                     'The number of samples per epoch.')
flags.DEFINE_integer('img_h', 60, 'The image height.')
flags.DEFINE_integer('img_w', 200, 'The image width.')
flags.DEFINE_integer('img_c', 3, 'The number of channels.')

def img_pre_processing(img, old = False):

    #if old:
    #    # resize and cast to float
    #    img = misc.imresize(
    #        img, (140, FLAGS.img_w)).astype('float')
    #else:
    #    # resize and cast to float
    #    img = misc.imresize(
    #        img, (100, FLAGS.img_w)).astype('float')
    #    img = img[40:]

    # normalize
    # img /= 255.
    # img -= 0.5
    #img *= 2.
    return img

def img_paths_to_img_array(image_paths):
    all_imgs = [misc.imread(imp) for imp in image_paths]
    return np.array(all_imgs, dtype='float')

def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')

def select_specific_set(iter_set):
    imgs, labs = [], []
    for _, row in iter_set:
        # extract the features and labels
        if row['center'] != 'center':
            img = img_pre_processing(misc.imread('data/'+row['center']))
            lab = row['angle']
            lab = lab.strip()
            lab = float(lab)
            
            if lab == 0:
                if np.random.random() < 0.1:
                    imgs.append(img)
                    labs.append(lab)
            
            # flip and append proportionally to steering angle
            #if np.random.random() < np.absolute(lab):
            else:
                #imgs.append(img)
                #labs.append(lab)
                img_f = np.fliplr(img)
                lab_f = -1 * lab
                imgs.append(img_f)
                labs.append(lab_f)
            
                fname = 'data/'+row['left']
                fname = fname.replace(" ","")
                img_left = img_pre_processing(misc.imread(fname))
                lab_left = row['angle']
                #lab_left = float(lab) + np.random.random()*0.1 + 0.2
                lab_left = float(lab) + 0.25
                imgs.append(img_left)
                labs.append(lab_left)
            
                img_f = np.fliplr(img_left)
                lab_f = -1 * lab_left
                imgs.append(img_f)
                labs.append(lab_f)
            
                fname = 'data/'+row['right']
                fname = fname.replace(" ","")
                img_right = img_pre_processing(misc.imread(fname))
                lab_right = row['angle']
                #lab_right = float(lab) - np.random.random()*0.1 - 0.2
                lab_right = float(lab) - 0.25
                imgs.append(img_right)
                labs.append(lab_right)

                img_f = np.fliplr(img_left)
                lab_f = -1 * lab_left
                imgs.append(img_f)
                labs.append(lab_f)
        
    return np.array(imgs), np.array(labs)

def generate_batch(log_data):
    while True:
        imgs, labs = select_specific_set(
            log_data.sample(
                FLAGS.batch_size).iterrows())
        yield np.array(imgs), np.array(labs)

def main(_):
    # fix random seed for reproducibility
    np.random.seed(123)

    # read the training driving log
    with open('data/driving_log.csv', 'rb') as f:
        log_data = pd.read_csv(
            f, header=None,
            names=['center', 'left', 'right', 'angle',
                   'throttle', 'break', 'speed'])
    print("Got", len(log_data), "samples for training")

    # read the validation driving log
    X_val, y_val = select_specific_set(
        log_data.sample(int(len(log_data)*.30)).iterrows())
    print("Got", len(X_val), "samples for validation")
    print(X_val.shape)
    print(y_val.shape)

    # create and train the model
    input_shape = (FLAGS.img_h, FLAGS.img_w, FLAGS.img_c)
    input_tensor = Input(shape=input_shape)

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5,
                     input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode="same", activation="relu"))
    #model.add(LeakyReLU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", activation="relu"))
    #model.add(LeakyReLU())
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same", activation="relu"))
    #model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dropout(.2))
    #model.add(LeakyReLU())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.2))
    #model.add(LeakyReLU())
    model.add(Dense(100, activation='relu'))
    #model.add(LeakyReLU())
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1))
    
    model.load_weights('model.h5')
    model.compile(optimizer="adam", loss="mse")
    
    history = model.fit_generator(
        generate_batch(log_data),
        samples_per_epoch=FLAGS.samples_per_epoch,
        validation_data=(X_val, y_val),
        nb_epoch=10,
        verbose=1)

    # save model to disk
    save_model(model)
    print('model saved')


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
