#!/usr/bin/env python

"""
train_classifier.py
-------------------------------------------------------------------------------------------------------------
Train a linear HOG-SVM vehicle classifier 
-------------------------------------------------------------------------------------------------------------
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

#------------------------------------------------------------------------------------------------------------
 
def bin_spatial(img, size=(32, 32)):
    """
    Function that computes spatial bin features.

    Args:
        img (numpy array): input image
        size (tuple): size to which the input image is resize
    Returns:
        features (numpy array): flattened feature vector
    """
    features = cv2.resize(img, size).ravel() 
    
    return features

#------------------------------------------------------------------------------------------------------------

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Function that computes color histograms of separate color channels as features.

    Args:
        img (numpy array): input image
        nbins (int): number of histogram bins
        bins_range (tuple): bin range
    Returns:
        features (numpy array): flattened feature vector
    """

    
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))

    return hist_features

#------------------------------------------------------------------------------------------------------------

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    """
    Function to get hog features of image.

    Args:
        img (numpy array): input image
        orient (int): number of orientations
        pix_per_cell (int): number of pixels per cell
        cell_per_block (int): number of cells per block
        vis (boolean): flag if set to True returns visualization of hog image
        feature_vec (boolean): flag if set to True returns hog feature as a vector
    Returns:
        features (numpy array): feature vector or array (depending on feature_vec flag)
        if vis is true:
            hog_image (numpy array): hog visualization of input image
    """

    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

#------------------------------------------------------------------------------------------------------------

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Function to extract features from a list of images.

    Args:
        imgs (list): list of images
        color_space (RGB, HSV, LUV, HLS, YUV, YCrCb): color space in which to extract features
        spatial_size (tuple): argument for bin_spatial()
        hist_bins (int): argument for color_hist()
        orient (int): argument for get_hog_features()
        pix_per_cell (int): argument for get_hog_features()
        cell_per_block (int): argument for get_hog_features()
        hog_channel (0,1,2,ALL): which or all channels to compute hog features
        spatial_feat (boolean): flag that determines if spatial bin features are computed
        hist_feat (boolean): flag that determines if color histogram features are computed
        hog_feat (boolean): flag that determines if hog features are computed
    Returns:
        features (list): list of features
    """

    
    features = []
    for file in imgs:
        file_features = []
        image = cv2.imread(file)
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        else: feature_image = np.copy(image)      

        #Compute spatial features
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        #Compute color histogram features
        if hist_feat == True:
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        #Compute hog features
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    return features

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    #Collect car and non-car images for training
    car_dirs = next(os.walk('vehicles/'))[1]
    cars = []
    for car_dir in car_dirs:
        car_images = glob.glob('vehicles/' + car_dir + '/' + '*.png')
        for image in car_images:
            cars.append(image)

    notcar_dirs = next(os.walk('non-vehicles/'))[1]
    notcars = []
    for notcar_dir in notcar_dirs:
        notcar_images = glob.glob('non-vehicles/' + notcar_dir + '/' + '*.png')
        for image in notcar_images:
            notcars.append(image)

    #Parameters for feature extraction
    color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off

    #Extract features for cars and non_cars
    t=time.time()
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)             
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.5, random_state=rand_state)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')

    #Train a Linear SVC
    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    
    # Save a dictionary containing the model and the model parameters
    model = {"svc":svc, "X_scaler": X_scaler, 
             "orient": orient, 
             "pix_per_cell": pix_per_cell, 
             "cell_per_block": cell_per_block,
             "spatial_size": spatial_size,
             "hist_bins": hist_bins}

    joblib.dump(model, 'Models/LinearSVC.pkl') 