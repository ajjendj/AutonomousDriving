#!/usr/bin/env python

"""
detect_vehicles.py
-------------------------------------------------------------------------------------------------------------
Loads vehicle classification model and detects vehicles with the following pipeline:
- Compute hog, spatial bin and color histogram features for lower half of each frame of input video
- Make predictions using sliding windows at multiple scales
- Compute heatmaps to combine overlapping vehicle detections
- Apply a threshold on the heatmap to filter non-confident detections
- Combine heatmaps of previous n frames to eliminate false positives
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
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

#------------------------------------------------------------------------------------------------------------

class VehicleDetector:
    """
    VehicleDetector class contains methods to find vehicles in input images.
    """
    def __init__(self, n):
        self.n = n #number of previous frames to keep in history
        self.last_n_heatmaps = [] #a list of the last n heatmaps 
        self.model = joblib.load('models/LinearSVC.pkl')
        self.fc = 0 #frame counter (useful in saving images)
        
    #------------------------------------------------------------------------------------------------------------
 
    def bin_spatial(self, img, size=(32, 32)):
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

    def color_hist(self, img, nbins=32, bins_range=(0, 256)):
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

    def convert_color(self, img, conv='BGR2HLS'):
        """
        Function to do colorspace conversions

        Args:
            img (numpy array): input image
            conv (RGB2YCrCb, BGR2YCrCb, RGB2LUV, RGB2HLS): 
        """
        if conv == 'RGB2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'BGR2YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if conv == 'BGR2LUV':
            return cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        if conv == 'BGR2HLS':
            return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        else:
            return img

    #------------------------------------------------------------------------------------------------------------

    def get_hog_features(self, img, orient, pix_per_cell, cell_per_block, 
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

        if vis == True:
            features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                      visualise=vis, feature_vector=feature_vec)
            return features, hog_image

        else:      
            features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                           cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                           visualise=vis, feature_vector=feature_vec)
            return features

    #------------------------------------------------------------------------------------------------------------

    def extract_features(self, imgs, cspace='RGB', orient=9, 
                            pix_per_cell=8, cell_per_block=2, hog_channel=0):
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
        # Iterate through the list of images
        for file in imgs:
            # Read in each one by one
            image = cv2.imread(file)
            # apply color conversion if other than 'RGB'
            if cspace != 'RGB':
                if cspace == 'HSV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                elif cspace == 'LUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
                elif cspace == 'HLS':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                elif cspace == 'YUV':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
                elif cspace == 'YCrCb':
                    feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: feature_image = np.copy(image)      

            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(self.get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)

            else:
                hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            features.append(hog_features)
        # Return list of feature vectors
        return features

    #------------------------------------------------------------------------------------------------------------

    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        """
        Function to draw boxes on the image to denote predicted vehicles.

        Args:
            img (numpy array): input image
            bboxes (list): list of bounding boxes
            color (3-tuple): color with which to draw bounding box
            thick (int): thickness of bounding box

        Returns:
            imcopy (numpy array): copy of input image with bounding boxes drawn
        """

        imcopy = np.copy(img)
        for bbox in bboxes:
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        return imcopy

    #------------------------------------------------------------------------------------------------------------

    def slide_window(self, img, x_start_stop=[None, None], y_start_stop=[None, None], 
                        xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
        """
        Function that returns a list of windows to be classified by the model.

        Args:
            x_start_stop (2 element list): start and end column coordinates
            y_start_stop (2 element list): start and end row coordinates
            xy_window (tuple): window size
            xy_overlap (tuple): overlap fraction in x and y directions
        Returns:
            window_list (list): list of window coordinates
        """
        
        if x_start_stop[0] == None:
            x_start_stop[0] = 0
        if x_start_stop[1] == None:
            x_start_stop[1] = img.shape[1]
        if y_start_stop[0] == None:
            y_start_stop[0] = 0
        if y_start_stop[1] == None:
            y_start_stop[1] = img.shape[0]

        # Compute the span of the region to be searched    
        xspan = x_start_stop[1] - x_start_stop[0]
        yspan = y_start_stop[1] - y_start_stop[0]
        # Compute the number of pixels per step in x/y
        nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
        ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
        # Compute the number of windows in x/y
        nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
        ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
        nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
        ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        for ys in range(ny_windows):
            for xs in range(nx_windows):
                # Calculate window position
                startx = xs*nx_pix_per_step + x_start_stop[0]
                endx = startx + xy_window[0]
                starty = ys*ny_pix_per_step + y_start_stop[0]
                endy = starty + xy_window[1]
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
        # Return the list of windows
        return window_list

    #------------------------------------------------------------------------------------------------------------

    def single_img_features(self, img, color_space='RGB', spatial_size=(32, 32),
                            hist_bins=32, orient=9, 
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True): 
        """
        Function to extract features from a single images.

        Args:
            imgs (numpy array): input image
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
            features (array): vector of concatentated features
        """   

        img_features = []
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        else: feature_image = np.copy(img)      
        
        #Compute spatial features if flag is set
        if spatial_feat == True:
            spatial_features = self.bin_spatial(feature_image, size=spatial_size)
            img_features.append(spatial_features)
        
        #Compute histogram features if flag is set
        if hist_feat == True:
            hist_features = self.color_hist(feature_image, nbins=hist_bins)
            img_features.append(hist_features)
        
        #Compute HOG features if flag is set
        if hog_feat == True:
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    feature = self.get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True)
                    hog_features.extend(feature)        
            else:
                hog_features = self.get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            hog_features = np.asarray(hog_features)
            img_features.append(hog_features)

        features = np.concatenate(img_features)
        return features

    #------------------------------------------------------------------------------------------------------------

    def search_windows(self, img, windows, clf, scaler, color_space='RGB', 
                        spatial_size=(32, 32), hist_bins=32, 
                        hist_range=(0, 256), orient=9, 
                        pix_per_cell=8, cell_per_block=2, 
                        hog_channel=0, spatial_feat=True, 
                        hist_feat=True, hog_feat=True):

        """
        Function where a list of windows in the input image is searched for vehicles.

        Args:
            imgs (numpy array): input image
            windows (list): list of windows outputted by slide_windows()
            clf (svc object): svc model
            scaler (scaler object): scaler
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
            on_windows (list): list of positive detection windows
        """

        on_windows = []
        for window in windows:
            #Extract the test window from original image and compute features for that window
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
            features = self.single_img_features(test_img, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
            #Scale extracted features to be fed to classifier
            test_features = scaler.transform(np.array(features).reshape(1, -1)) 
            #Predict using your classifier
            prediction = clf.predict(test_features)
            #If positive, then save the window
            if prediction == 1:
                on_windows.append(window)
        #Return list of windows corresponding to positive detections
        return on_windows

    #------------------------------------------------------------------------------------------------------------

    def add_heat(self, heatmap, bbox_list):
        """
        Function to compute heatmap based on list of bounding boxes of positive detections.
        
        Args:
            heatmap (numpy array): An array of size equal to the input image initialized to zeros_like
            bbox_list (list): list of bounding boxes with positive detections.
        Returns:
            heatmap (numpy array): Heatmap of detections
        """

        for box in bbox_list:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        #Normalize heatmap by the max value
        heatmap = heatmap/(np.max(heatmap) + 1e-16)
        return heatmap

    #------------------------------------------------------------------------------------------------------------
        
    def apply_threshold(self, heatmap, threshold):
        """
        Function to apply threshold to heatmap.

        Args:
            heatmap (numpy array): input heatmap
            threshold (float): threshold
        Returns:
            heatmap (numpy array): output heatmap with all pixels below threshold set to zeros_like
        """

        heatmap[heatmap <= threshold] = 0
        return heatmap

    #------------------------------------------------------------------------------------------------------------

    def draw_labeled_bboxes(self, img, labels):
        """
        Funtion to draw bounding boxes on detected vehicles/

        Args:
            img (numpy array): input image
            labels (list): list of detected vehicles
        Returns:
            img (numpy array): input image with bounding boxes drawn
        """

        for car_number in range(1, labels[1]+1):
            #Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            #Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            #Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            #Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        #Return the image
        return img

    #------------------------------------------------------------------------------------------------------------

    def find_cars(self, img, ystart, ystop, color_space, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, bbox_list = []):
        """
        Function that extract features from a single image using hog sub-sampling and makes predictions

        Args:
            img (numpy array): input image
            bbox_list (list): list of bounding boxes (initially initialized to empty)
            ystart (int): row coordinate from which to start searching for vehicles
            ystop (int): row coordinate up to which to search for vehicle
            scale (float): determines how many scales of windows to use to search for vehicles
            svc (svc object): svc model
            X_scaler (scaler object): scaler
            orient (int): argument for get_hog_features()
            pix_per_cell (int): argument for get_hog_features()
            cell_per_block (int): argument for get_hog_features()
        Returns:
            draw_img (numpy array): input image with bounding boxes drawn
            bbox_list (list): list of bounding boxes corresponding to windows where vehicles have been predicted
        """
        
        draw_img = np.copy(img)
        xstart = 0
        img_tosearch = img[ystart:ystop,xstart:,:]
        ctrans_tosearch = self.convert_color(img_tosearch, conv='BGR2HLS')
        #img_tosearch = img_tosearch.astype(np.float32)/255
        
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]

        nxblocks = (ch1.shape[1] // pix_per_cell)-1
        nyblocks = (ch1.shape[0] // pix_per_cell)-1 
        nfeat_per_block = orient*cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell)-1 
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = self.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = self.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = self.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*pix_per_cell 
                ytop = ypos*pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
                spatial_features = self.bin_spatial(subimg, size=spatial_size)
                hist_features = self.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                #test_features = X_scaler.transform((hog_features).reshape(1, -1))    
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                test_prediction = svc.predict(test_features)

                thresh = 0.5
                conf = svc.decision_function(test_features)
                prediction = int(conf > thresh)

                if (prediction == 1): # and ((np.exp(test_prob[0][1])) > thresh):
                    xbox_left = np.int(xleft*scale) + xstart
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                    bbox = ((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                    bbox_list.append(bbox)
                    
        return draw_img, bbox_list

    #------------------------------------------------------------------------------------------------------------

    def detect_vehicles(self, image, ystart=360, ystop=656, scales = [1,1.5,2]):
        """
        Function that takes an input image and detects vehicles on it by calling the find_cars function.

        Args: 
            image (numpy array): input array
            ystart (int): row coordinate from which to start searching for vehicles
            ystop (int): row coordinate up to which to search for vehicle 
            
        Returns:
            out_image (numpy array): output image with bounding boxes drawn
        """

        model = self.model
        svc = model["svc"]
        X_scaler = model["X_scaler"]
        orient = model["orient"]
        pix_per_cell = model["pix_per_cell"]
        cell_per_block = model["cell_per_block"]
        spatial_size = model["spatial_size"]
        hist_bins = model["hist_bins"]

        #Note: this function is passed to fl_image function which reads images in RGB.
        #Because the classifier was trained on BGR images (using cv2.imread()), make necessary conversion
        imageBGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        color_space = 'HLS'
        box_list = []
        out_image1, box_list = vd.find_cars(imageBGR, ystart, ystop, color_space, scales[0], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, box_list)
        out_image2, box_list = vd.find_cars(imageBGR, ystart, ystop, color_space, scales[1], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, box_list)
        out_image3, box_list = vd.find_cars(imageBGR, ystart, ystop, color_space, scales[2], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, box_list)
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = self.add_heat(heat,box_list)
        height, width = image.shape[:2]
        self.last_n_heatmaps.append(heat)
        if len(self.last_n_heatmaps) > self.n:
            self.last_n_heatmaps.pop(0)
        cumulative_heat_maps = np.sum(np.asarray(self.last_n_heatmaps),0)/len(self.last_n_heatmaps)
        heatmap = cumulative_heat_maps * 255.0
        # Apply threshold to help remove false positives
        heat = self.apply_threshold(cumulative_heat_maps,0.5)
        # Find final boxes from heatmap using label function
        labels = label(heat)
        out_image = self.draw_labeled_bboxes(np.copy(image), labels)
        self.fc += 1

        return out_image

    #------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------

def test_images(path_to_test_dir, path_to_output_dir):
    """
    Function that runs the vehicle detection pipeline on a test directory of images.

    Args:
        path_to_test_dir: path to test directory
        path_to_output_dir: path to output directory where images are saved
    """

    # Get list of test images
    images = glob.glob(path_to_test_dir + '/*.jpg')

    # For each test image
    for image_name in images:
        print (image_name)
        image = cv2.imread(image_name)
        draw_image = np.copy(image)
        vd = VehicleDetector(1)
        # Load model and model parameters
        color_space = 'HLS'
        hog_channel='ALL'
        spatial_feat=True
        hist_feat=True
        hog_feat=True
        ystart= 400
        ystop=656
        scales=[1,1.5,2]
        model = vd.model
        svc = model["svc"]
        X_scaler = model["X_scaler"]
        orient = model["orient"]
        pix_per_cell = model["pix_per_cell"]
        cell_per_block = model["cell_per_block"]
        spatial_size = model["spatial_size"]
        hist_bins = model["hist_bins"]

        windows1 = vd.slide_window(image, x_start_stop=[None, None], y_start_stop=[ystart,ystop], 
                        xy_window=(96, 96), xy_overlap=(0.7, 0.7))
        windows2 = vd.slide_window(image, x_start_stop=[None, None], y_start_stop=[ystart,ystop-75], 
                        xy_window=(64, 64), xy_overlap=(0.6, 0.6))
        windows3 = vd.slide_window(image, x_start_stop=[None, None], y_start_stop=[ystart,ystop-150], 
                        xy_window=(48, 48), xy_overlap=(0.5, 0.5))

        windows = windows1 + windows2 + windows3 
        hot_windows = vd.search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       

        window_img = vd.draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                   
        cv2.imwrite(path_to_output_dir + '/' + image_name[12:-4] + '_A_window.jpg', window_img)

        

        box_list = []
        out_image1, box_list = vd.find_cars(image, ystart, ystop - 150, color_space, scales[0], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, box_list)
        out_image2, box_list = vd.find_cars(image, ystart, ystop - 75, color_space, scales[1], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, box_list)
        out_image3, box_list = vd.find_cars(image, ystart, ystop, color_space, scales[2], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, box_list)
        
        out_imageALL = np.copy(image)
        for i in range(len(box_list)):
            cv2.rectangle(out_imageALL, (box_list[i][0][0], box_list[i][0][1]),(box_list[i][1][0], box_list[i][1][1]),(0,0,255),6)

        height, width = image.shape[:2]
        cv2.imwrite(path_to_output_dir + '/' + image_name[12:-4] + '_original.jpg', cv2.resize(image, (int(0.25*width), int(0.25*height))))
        cv2.imwrite(path_to_output_dir + '/' + image_name[12:-4] + '_B_alldetections1.jpg', cv2.resize(out_image1, (int(0.25*width), int(0.25*height))))
        cv2.imwrite(path_to_output_dir + '/' + image_name[12:-4] + '_B_alldetections2.jpg', cv2.resize(out_image2, (int(0.25*width), int(0.25*height))))
        cv2.imwrite(path_to_output_dir + '/' + image_name[12:-4] + '_B_alldetections3.jpg', cv2.resize(out_image3, (int(0.25*width), int(0.25*height))))
        cv2.imwrite(path_to_output_dir + '/' + image_name[12:-4] + '_B_alldetectionsALL.jpg', cv2.resize(out_imageALL, (int(0.25*width), int(0.25*height))))
        
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = vd.add_heat(heat,box_list)
        heatmap = heat * 255.0
        heat = vd.apply_threshold(heat,0.5)
        labels = label(heat)
        draw_img = vd.draw_labeled_bboxes(np.copy(draw_image), labels)
        cv2.imwrite(path_to_output_dir + '/' + image_name[12:-4] + '_C_heatmap.jpg', cv2.resize(heatmap, (int(0.25*width), int(0.25*height))))
        cv2.imwrite(path_to_output_dir + '/' + image_name[12:-4] + '_D_final.jpg', cv2.resize(draw_img, (int(0.25*width), int(0.25*height))))

    return

#------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    ##Run pipeline on test images
    #test_images('test_images', 'output_images')

    #Run pipeline on test/project video
    vd = VehicleDetector(5)
    output = 'test_video_out.mp4'
    clip = VideoFileClip('test_video.mp4')
    output_clip = clip.fl_image(vd.detect_vehicles)
    output_clip.write_videofile(output, audio=False)
