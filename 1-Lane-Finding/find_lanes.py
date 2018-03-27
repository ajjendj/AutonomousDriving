#!/usr/bin/env python

"""
find_lanes.py
-------------------------------------------------------------------------------------------------------------
Loads camera calibation parameters and finds lanes with the following pipeline:
- Undistort input image
- Create a thresholded binary image
- Apply a perspective transform to rectify the binary image
- Detect lane pixels and fits to find the lane boundary
- Determine the curvature of the lane and vehicle position with respect to the center
- Warp detected lane boundaries back onto the original image

-------------------------------------------------------------------------------------------------------------
"""

import numpy as np
import cv2
import glob
import pickle
from moviepy.editor import VideoFileClip

#------------------------------------------------------------------------------------------------------------

class LaneFinder:
    """
    LaneFinder class contains methods to find lanes from input images.
    """
    def __init__(self, n):
        self.n = n #number of previous frames to keep in history
        self.last_n_plf = [] #a list of poly1d objects for the left lane
        self.last_n_prf = [] #a list of poly1d objects for the right lane
        self.last_left_fit = [] #the most recent polyfit for the left lane
        self.last_right_fit = [] #the most recent polyfit for the right lane
        self.diff_thresh = 50 #threshold to determine if lanes are outliers (unit pixels) 
        self.num_outliers = 0 #count of number of successive outliers
        self.is_prev_outlier = False #boolean to track wheter previous frame was outlier

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """
        Function that applies the Sobel edge detector in a particular direction.

        Args:
            img (numpy array): input image.
            orient (string): 'x' or 'y' direction.
            sobel_kernel (odd int): size of kernel.
            thresh (tuple): lower and upper threshold to apply to edge image.

        Returns:
            binary_output (numpy array): thresholded Sobel-edge image
        """

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

        return binary_output

    #------------------------------------------------------------------------------------------------------------

    def mag_threshold(self, img, sobel_kernel=3, thresh=(0, 255)):
        """
        Function that applies the Sobel edge detector and a threshold on the edge magnitudes.

        Args:
            img (numpy array): input image.
            sobel_kernel (odd int): size of kernel.
            thresh (tuple): lower and upper threshold to apply on gradient magnitudes.

        Returns:
            binary_output: thresholded Sobel-edge image
        """

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 255

        return binary_output

    #------------------------------------------------------------------------------------------------------------

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        """
        Function that applies the Sobel edge detector and a threshold on the edge orientations.

        Args:
            img (numpy array): input image.
            sobel_kernel (odd int): size of kernel.
            thresh (tuple): lower and upper threshold to apply on gradient orientations.

        Returns:
            binary_output: thresholded Sobel-edge image
        """
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 255
        
        return binary_output

    #------------------------------------------------------------------------------------------------------------

    def hls_select(self, img, thresh=(0, 255)):
        """
        Function that thresholds an image based on s channel of an hsv image.

        Args:
            img (numpy array): input image.
            thresh (tuple): lower and upper threshold to apply on s values.

        Returns:
            binary_output: thresholded s-channel image
        """

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:,:,2]
        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 255
        return binary_output

    #------------------------------------------------------------------------------------------------------------

    def filter_by_edgeandcolor(self, image, ksize=3, edge_thresh=(20,100), mag_thresh=(70,200), 
                               dir_thresh=(np.pi/3,np.pi - np.pi/3), col_thresh=(100,255)):
        """
        Function that applies several filters and produces corresponding binary images, which
        are combined to produce a binary image in which the lanes are present.

        Args:
            img (numpy array): input image.
            ksize (odd int): size of sobel kernel
            edge_thresh (tuple): lower and upper threshold to apply to edge image.
            dir_thresh (tuple): lower and upper threshold to apply to gradient orientations.
            mag_thresh (tuple): lower and upper threshold to apply on gradient magnitudes.
            col_thresh (tuple): lower and upper threshold to apply on s values of hsv image.

        Returns:
            int_output (list of numpy arrays): list of outputs of filters
            combined (numpy array): combined image
        """

        gradx = self.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=edge_thresh)
        grady = self.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=edge_thresh)
        mag_binary = self.mag_threshold(image, sobel_kernel=ksize, thresh=mag_thresh)
        dir_binary = self.dir_threshold(image, sobel_kernel=ksize, thresh=dir_thresh)
        col_binary = self.hls_select(image, thresh=col_thresh)

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 255) & (grady == 255)) | ((mag_binary == 255) & (dir_binary == 255)) | (col_binary == 255)] = 255

        return [gradx, grady, mag_binary, dir_binary, col_binary], combined

    #------------------------------------------------------------------------------------------------------------

    def get_birdseyeview(self, persp_transf_src, persp_transf_dst, RGB, binary):
        """
        Gets a birdeyeview of the image by computing a perspective transform and applying it to the input images.

        Args:
            persp_transf_src (numpy array) : source array of trapezoid
            ersp_transf_dst (numpy array) : destination array of rectangle
            RGB (numpy array): source RGB image
            binary (numpy array): source binary image

        Returns:
            perspective_mat (numpy array): perspective transform matrix
            warped_RGB (numpy array): warped RGB image
            warped_binary (numpy array): warped binary image
        """
        img_size = (binary.shape[1], binary.shape[0])
        perspective_mat = cv2.getPerspectiveTransform(persp_transf_src, persp_transf_dst)
        warped_RGB = cv2.warpPerspective(RGB,perspective_mat,img_size)
        warped_binary = cv2.warpPerspective(binary,perspective_mat,img_size)

        return perspective_mat, warped_RGB, warped_binary

    #------------------------------------------------------------------------------------------------------------

    def get_polyline(self, binary_warped):
        """
        Fits 2-D polynomials, one each for the left and right lanes.

        Args:
            binary (numpy array) : source warped binary array

        Returns:
            lane indices (list) : list of leftx, lefty, rightx, righty lane pixel indices
            out_img (numpy array) : array with detected rectangles drawn
        """

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set the width of the windows +/- margin
        margin = 100

        if not(self.last_n_plf and self.last_n_prf):

            # Compute histograms to find peaks corresponding to lanes (in bottom 2/3ds of image)
            histogram = np.sum(binary_warped[int(binary_warped.shape[0]/1.5):,:], axis=0)
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            midpoint = np.int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Find lane-windows across nwindows sliding windows
            nwindows = 9
            window_height = np.int(binary_warped.shape[0]/nwindows)
            
            leftx_current = leftx_base
            rightx_current = rightx_base

            # Set minimum number of pixels found to recenter window
            minpix = 50
            # Create empty lists to receive left and right lane pixel indices
            left_lane_inds = []
            right_lane_inds = []

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
                # Identify the nonzero pixels in x and y within the window
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                # Append these indices to the lists
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:        
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

            # Concatenate the arrays of indices
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            plf = np.poly1d(left_fit)
            prf = np.poly1d(right_fit)

            self.last_n_plf.append(plf)
            self.last_n_prf.append(prf)
            self.last_left_fit.append(left_fit)
            self.last_right_fit.append(right_fit)

        else:

            left_fit = self.last_left_fit[0]
            right_fit = self.last_right_fit[0]
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            # Instantiate polynomial class
            plf = np.poly1d(left_fit)
            prf = np.poly1d(right_fit)

            # Generate x and y values for plotting
            left_line_window1 = []
            left_line_window2 = []
            right_line_window1 = []
            right_line_window2 = []

            for row in range(binary_warped.shape[0]):
                left_line_window1.append([plf(row) - margin, row])
                left_line_window2.append([plf(row) + margin, row])
                right_line_window1.append([prf(row) - margin, row])
                right_line_window2.append([prf(row) + margin, row])

            left_line_pts = np.hstack((np.asarray(left_line_window1), np.asarray(left_line_window2)))
            right_line_pts = np.hstack((np.asarray(right_line_window1), np.asarray(right_line_window2)))

            # Create an image to draw on and an image to show the selection window
            out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
            window_img = np.zeros_like(out_img)
            # Color in left and right line pixels
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, [np.int32(left_line_pts).reshape((-1,1,2))], (0,255, 0))
            cv2.fillPoly(window_img, [np.int32(right_line_pts).reshape((-1,1,2))], (0,255, 0))
            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Compute similarity with previous frames' polynomials to detect outliers
            left_x_prev = np.zeros(binary_warped.shape[0])
            right_x_prev = np.zeros(binary_warped.shape[0])
            left_x_curr = np.zeros(binary_warped.shape[0])
            right_x_curr = np.zeros(binary_warped.shape[0])
            for row in range(binary_warped.shape[0]):
                left_x_prev[row] = self.last_n_plf[-1](row)
                right_x_prev[row] = self.last_n_prf[-1](row)
                left_x_curr[row] = plf(row)
                right_x_curr[row] = prf(row)

            left_diff = np.amax(np.absolute(left_x_prev - left_x_curr))
            right_diff = np.amax(np.absolute(right_x_prev - right_x_curr))

            if left_diff < self.diff_thresh and right_diff < self.diff_thresh:
                self.last_n_plf.append(plf)
                self.last_n_prf.append(prf)
                self.last_left_fit.pop()
                self.last_right_fit.pop()
                self.last_left_fit.append(left_fit)
                self.last_right_fit.append(right_fit)
                self.is_prev_outlier = False
                self.num_outliers = 0
            else:
                self.is_prev_outlier = True
                self.num_outliers += 1

            if len(self.last_n_plf) > self.n:
                self.last_n_plf.pop(0)
                self.last_n_prf.pop(0)

            # If number of consecutive outliers > n, reset
            if self.num_outliers > self.n:
                self.last_n_plf[:] = []
                self.last_n_prf[:] = []
                self.last_left_fit[:] = []
                self.last_right_fit[:] = []
                self.num_outliers = 0

            out_img = cv2.resize(out_img,None,fx=.25, fy=.25, interpolation = cv2.INTER_CUBIC)
            cv2.imwrite('test_out.jpg', out_img)

        return [leftx, lefty, rightx, righty], out_img

    #------------------------------------------------------------------------------------------------------------

    def map_lane(self, left_lane_points, right_lane_points, perspective_mat, undistorted):
        """
        Maps the lane onto the original image.

        Args:
            left_lane_points (numpy array): points for the left_lane boundaries
            right_lane_points (numpy array): points for the right_lane boundaries
            perspective_mat (numpy array): the perspective transform matrix
            undistorted (numpy array): original image on which to draw lane 

        Returns:
            lanelines (numpy array): original image overlayed with lane
        """

        color_warp = np.zeros_like(undistorted).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array(left_lane_points)
        pts_right = np.array(np.flipud(right_lane_points))
        pts = np.vstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        Minv = np.linalg.inv(perspective_mat)
        newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
        # Combine the result with the original image
        lanelines = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

        return lanelines

    #------------------------------------------------------------------------------------------------------------

    def compute_curvature(self, line_pixels):
        '''
        Computes the curvature of the lane.

        Args:
            line_pixels (list): list containing x,y indices of left and right lane pixels

        Returns:
            left_curverad (float): left lane curvature in meters
            right_curverad (float): right lane curvature in meters
        '''

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        y_eval = 720

        leftx, lefty, rightx, righty = line_pixels
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        return left_curverad, right_curverad

    #------------------------------------------------------------------------------------------------------------

    def compute_offset(self, line_pixels):
        '''
        Computes the offset from the center of the lane.

        Args:
            line_pixels (list): list containing x,y indices of left and right lane pixels

        Returns:
            offset (float): offset in meters
        '''

        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        center_camera = 640 #image is 1280 pixels wide
        leftx, lefty, rightx, righty = line_pixels

        #get x-index of left and right lanes
        left_bottom = leftx[0]
        right_bottom = rightx[0]

        #compute offset
        center_lane = (left_bottom + right_bottom)/2
        offset = (center_camera - center_lane) * xm_per_pix
        return offset

    #------------------------------------------------------------------------------------------------------------

    def pipeline(self, image, image_name = '', write_images = False):
        """
        Function that implements the lane detection pipeline:
        1) undistort 
        2) threshold
        3) define trapezoid to get bird's eye view
        4) use trapezoid to warp image using perspective transform
        5) identify lane lines
        6) compute radius of curvature and offset
        7) map lanes back to original image

        Args:
            image (numpy array): input image on which to detect lane.
            image_name (string): input image name
            mtx (numpy array): camera calibration matrix
            dist (numpy array): undistortion coefficients
            write_images (boolean): whether to write the output images
            
        Return:
            final_image (numpy array): returns final image with detected lanes, curvature and offset
        """

        # Read in the saved camera matrix and distortion coefficients
        dist_pickle = pickle.load( open( "camera_cal/camera_calibration.p", "rb" ) )
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]

        # 1)
        undistorted = cv2.undistort(image, mtx, dist, None, mtx)

        # 2)
        intermediate_binary_list, thresholded = self.filter_by_edgeandcolor(undistorted)

        # 3)
        img_size = (1280, 720)
        persp_transf_src = np.float32([[(img_size[0] / 2) - 62, img_size[1] / 2 + 100], 
        [((img_size[0] / 6) - 10), img_size[1]],
        [(img_size[0] * 5 / 6) + 60, img_size[1]],
        [(img_size[0] / 2 + 62), img_size[1] / 2 + 100]])
        persp_transf_dst = np.float32([[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])

        trapezoid = undistorted.copy()
        cv2.polylines(trapezoid, [np.int32(persp_transf_src).reshape((-1,1,2))], True, (0,0,255),1)

        # 4)
        perspective_mat, warped_RGB, warped_binary = self.get_birdseyeview(persp_transf_src, persp_transf_dst, trapezoid, thresholded)

        # 5)
        binary_warped = warped_binary/255
        line_pixels, out_img = self.get_polyline(binary_warped)

        # Generate x and y values for plotting
        left_lane_points = []
        right_lane_points = []

        # For each row (i.e y value)
        for row in range(binary_warped.shape[0]):
            left_x = []
            right_x = []
            num = len(self.last_n_plf)

            # Take the weighted average of plf(row) of last n frames for both left and right frames
            for n in range(len(self.last_n_plf)):
                left_x.append(self.last_n_plf[n](row) * (n+1))
                right_x.append(self.last_n_prf[n](row) * (n+1))
            left = (np.sum(np.asarray(left_x)))/((num*num + num)/2)
            right = (np.sum(np.asarray(right_x)))/((num*num + num)/2)

            # Append (x,y) pixel points for the left and right lanes
            left_lane_points.append([left, row])
            right_lane_points.append([right, row])

        left_lane_points = np.asarray(left_lane_points)
        right_lane_points = np.asarray(right_lane_points)

        cv2.polylines(out_img, [np.int32(left_lane_points).reshape((-1,1,2))], False, (0,0,255),3)
        cv2.polylines(out_img, [np.int32(right_lane_points).reshape((-1,1,2))], False, (0,0,255),3)

        # 6)
        left_curverad, right_curverad = self.compute_curvature(line_pixels)
        offset = self.compute_offset(line_pixels)

        # 7)
        lanelines = self.map_lane(left_lane_points, right_lane_points, perspective_mat, undistorted)

        # print curvature and offset
        font = cv2.FONT_HERSHEY_SIMPLEX
        lanelines_withtext = lanelines.copy()
        curverad = np.mean([left_curverad, right_curverad])
        curvature_text = 'Curvature: {:.2f} meters'.format(curverad)
        offset_text = ''
        if offset < 0:
            offset_text = 'Offset: {:.2f} meters to the left'.format(np.absolute(offset))
        elif offset > 0:
            offset_text = 'Offset: {:.2f} meters to the right'.format(np.absolute(offset))
        else:
            offset_text = 'Offset: dead center'.format(np.absolute(offset))

        cv2.putText(lanelines_withtext, curvature_text, (460,600), font, 0.9,(255,0,0),2)
        cv2.putText(lanelines_withtext, offset_text, (460,650), font, 0.9,(255,0,0),2)

        pipeline_images = [undistorted,thresholded,trapezoid,warped_RGB,warped_binary,out_img,lanelines,lanelines_withtext]

        if write_images==True:
            backslash_ind = image_name.index('/')
            full_image_name = image_name[backslash_ind+1:-4]

            # Write images from pipeline to disk
            self.write_pipeline_images(pipeline_images, full_image_name, resize=True)

        final_image = pipeline_images[-1]

        return final_image

    #------------------------------------------------------------------------------------------------------------

    def write_pipeline_images(self, pipeline_images, image_name, output_dir='output_images/', resize=False):
        """
        Function that writes all images corresponding to the steps of the lane detection pipeline.

        Args:
            pipeline_images (list of numpy arrays): list of pipeline images.
            output_dir (string): output directory path/name
            image_name (string): name of input name which will be used as a prefix

        """

        pipeline = ['_undistorted',
                    '_thresholded',
                    '_trapezoid',
                    '_warped_RGB',
                    '_warped_binary',
                    '_lanelines_binary',
                    '_lanelines',
                    '_lanelineswithtext']

        for i in range(len(pipeline_images)):
            output_name = output_dir + image_name + '_' + str(i) + '_' + pipeline[i] + '.jpg'
            image = pipeline_images[i]
            if resize:
                image = cv2.resize(image,None,fx=.25, fy=.25, interpolation = cv2.INTER_CUBIC)
                output_name = output_dir + 'small/' + image_name + '_' + str(i) + '_' + pipeline[i] + '.jpg'
            cv2.imwrite(output_name, image)

        return

#------------------------------------------------------------------------------------------------------------

def test_images(path_to_test_dir, path_to_output_dir):
    """
    Function that runs the pipeline on a test directory of images.

    Args:
        test_dir: path to test directory
        output_dir: path to output directory where images are saved
    """

    # Get list of test images
    images = glob.glob(path_to_test_dir + '/*.jpg')

    # For each test image
    for image_name in images:
        print (image_name)
        input_image = cv2.imread(image_name)

        # Run lane detection pipeline
        fl = FindLanes()
        lane_image = fl.pipeline(input_image, image_name, write_images = True)

#------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    #test_images('test_images/', 'output_images/')

    fl = LaneFinder(10)
    output = 'project_video_out.mp4'
    clip = VideoFileClip('project_video.mp4')
    output_clip = clip.fl_image(fl.pipeline)
    output_clip.write_videofile(output, audio=False)

   




