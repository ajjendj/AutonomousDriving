#**Advanced Lane Finding Project**

The goals of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images captured by a camera.
* Apply a distortion correction to raw images.
* Use color and gradient information to create a thresholded binary image.
* Apply a perspective transform to rectify binary image to create a "birds-eye view".
* Detect lane pixels and fit a degree-2 polynomial to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

####1. Submission includes all required files.

| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `camera_calibration.py`      | Computes camera calibration parameters from set of chessboard images.              |
| `find_lanes.py`              | Contains the lane detection pipeline functions.             		        		|
| `output_images`              | Folder containing output images depicting intermediate to final results from pipeline. |
| `project_video_out.py`       | Video of the output of the lane detection pipeline applied to the example video.   |  
| `writeup.md`                 | Document summarizing the project 													|


[//]: # (Image References)

[image0]: ./output_images/small/chessboard1.jpg "Chessboard Original"
[image1]: ./output_images/small/undistorted_chessboard1.jpg "Chessboard"
[image2]: ./output_images/small/test6.jpg "Road"
[image3]: ./output_images/small/test6_0__undistorted.jpg "Road Undistorted"
[image4]: ./output_images/small/test6_1__thresholded.jpg "Binary Example"
[image5]: ./output_images/small/test6_2__trapezoid.jpg "Trapezoid Example"
[image6]: ./output_images/small/test6_3__warped_RGB.jpg "Warp RGB Example"
[image7]: ./output_images/small/test6_4__warped_binary.jpg "Warp binary Example"
[image8]: ./output_images/small/test6_5__lanelines_binary.jpg "Fit Visual"
[image9]: ./output_images/small/test6_6__lanelines.jpg "Output"
[image10]: ./output_images/small/test6_7__lanelineswithtext.jpg "Output with Curvature/Offset"
[image11]: ./output_images/small/lookahead.jpg "Output with Curvature/Offset"
[video1]: ./project_video_out.mp4 "Video"

####2. Camera Calibration

Camera calibration is implemented in `camera_calibration.py`. For each chessboard image in the provided camera_cal folder, the "object points", which are the (x, y, z) world coordinates of the chessboard corners, are prepared (assuming the chessboard is fixed on the z = 0 plane). The "image points" are the (x, y) pixel coordinates of the chessboard corners. The accumulated corresponding "objpoints" and "imgpoints" are then used to compute the camera calibration matrix along with the distortion coefficients by using the `cv2.calibrateCamera()` function. Applying distortion correction to one of the chessboard images yields the following result:

![alt text][image0]
![alt text][image1]

####3. Lane Detection Pipeline

Lane detection, implemented in the `pipeline()` function in `find_lanes.py` consists of the following steps:

#####a. Distortion Correction

The camera calibration and distortion correction parameters computed previously is used to undistort the original image captured by the camera placed on the car. A test image before and after distortion correction looks like this (Although hard to discern, notice the distance between the white car on the right to the image border):

![alt text][image2]
![alt text][image3]

#####b. Thresholding

The distortion-corrected image is then thresholded using a combination of edge and color filters. The edge filters outputs binary images created by applying thresholds to a) Sobel-edge images in the x and y directions, b) Sobel-edge magnitudes, c) Sobel-edge orientations. The color filter outputs a binary image created by applying thresholds to the 's' channel of an hls images (converted from BGR). These multiple binary maps are then combined in the `filter_by_edgeandcolor()` function in `find_lanes.py`. The final binary image looks like this:

![alt text][image3]
![alt text][image4]

#####c. Perspective Transform

The thresholded image is then perspective-transformed in order to get a birds-eye view. A trapezoid (along with the rectangle to which the trapezoid was to be transformed) was hardcoded. This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 578, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 702, 460      | 960, 0        |

The trapezoid can be seen in the following figure:

![alt text][image5]

A perspective transformation matrix between the trapezoid and rectangle is computed and then applied to the binary image, implemented in the `get_birdseyeview()` function in `find_lanes.py`, (as well as RGB image for visualization):

![alt text][image6]
![alt text][image7]

Note: The lanes here do not seem parallel due to the curvature of the lanes. For straight lanes, the lanes in the warped image appear to be parallel.

#####d. Lane Detection

Finally, the lanes are detected in the warped binary image. First, the lane positions at the bottom of the image is determined with the help of histogram peaks in the bottom 2/3rds of the image. Then, rectangles are slided along 9 horizontal layers to determine the optimal positions of the left and right lanes. The pixels that fall inside these rectangles are then used to fit a 2-D polynomial function to represent straight as well as curved lanes, implemented in the `get_polyline()` function in `find_lanes.py`. A visualization of the lane-detected rectangles as well as lanes can be seen here:

![alt text][image7]
![alt text][image8]

The detected lines are then projected back to the original image using the inverse perspective matrix:

![alt text][image9]

#####e. Curvature and Offset

The radius of curvature and offset from center are computed in the `compute_curvature` and `compute_offset` functions respectively. The mean curvature (of the two lines) and offset is then printed on the image like so:

![alt text][image10]

####4. Lane Detection Pipeline for Videos

The lane detection pipeline was then implemented for videos by encapsulating the above methods into a `LaneFinder` class. For every frame, the lanes, curvature and offset are computed. In every subsequent frame after the lanes have first been detected, the lanes are searched only around a window to the left and right margins of the lines detected in the most recent frame (i.e. a full left-to-right search is not conducted). 

![alt text][image11]

A weighted average of the lane polynomials of the previous 'n' frames is used to compute and draw the left and right lanes to smooth any jerky behavior. Furthermore, outliers are detected by comparing the polynomial of the current frame with that of the previous frame. If outliers are detected, they are ignored and lane polynomials of the previous frame are used to draw the lanes on the current frame. 

Here's a [link to my video result](https://vimeo.com/219456430)

####5. Discussion

The pipeline to find lanes depends on the robustness of the binarizing algorithm. In areas where the gradient and color signal are weak (for example, in the project video, there is a section on the bridge where the road is light and therefore it is harder to distinguish the lanes), the algorithm has some failures. Propagating lane properties from a window of n previous frames helps alleviate the effects of outliers. However, for a longer sequence of frames where the lanes are harder to detect (e.g. when the camera is under direct illumination), the algorithm will likely fail. Because pipeline approaches have several stages where the algorithm can fail, perhaps an end-to-end model trained on a large and diverse dataset will be more robust

