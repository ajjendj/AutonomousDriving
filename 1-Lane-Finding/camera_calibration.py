"""
camera_calibration.py
-------------------------------------------------------------------------------------------------------------
Computes camera calibation parameters, so that images captured from the camera can be undistorted. 
-------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import cv2
import glob
import pickle

# prepare object points
nx = 9
ny = 6

#Arrays to store object points and image points from all images
objpoints = []
imgpoints = []

#Prepare object points, like (0,0,0), (1,0,0), ..., (8,6,0)
objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #x,y coordinates

# Read in a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

for fname in images:
	# Read in each image
	img = cv2.imread(fname)

	# Convert to grayscale
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray_img, (nx, ny), None)

	# If found, append image and object points
	if ret == True:
		imgpoints.append(corners)
		objpoints.append(objp)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1], None, None)

example_image = cv2.imread('camera_cal/calibration2.jpg')
undistorted = cv2.undistort(example_image, mtx, dist, None, mtx)
undistorted_small = cv2.resize(undistorted,None,fx=.25, fy=.25, interpolation = cv2.INTER_CUBIC)
cv2.imwrite('output_images/small/undistorted_chessboard2.jpg', undistorted_small)

dist_pickle = { "mtx": mtx, "dist": dist}
pickle.dump(dist_pickle, open( "camera_cal/camera_calibration.p", "wb" ))    

