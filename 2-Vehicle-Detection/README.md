#**Vehicle Detection Project**

The goals this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Augment the HOG feature vector with spatial bins and color histogram features and train a Linear SVM classifier.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the vehicle detection pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

####1. Submission includes all required files.

| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `train_classifier.py`        | Computes feature vectors and trains a Linear SVM vechicle classifier.              |
| `detect_vehicles.py`         | Contains the vehicle detection pipeline functions.             		      		|
| `models`              	   | Folder containing saved models. 													|
| `output_images`              | Folder containing output images depicting intermediate to final results from pipeline. |
| `example_images`             | Folder containing example images used in this write-up. 							|
| `project_video_out.py`       | Video of the output of the lane detection pipeline applied to the example video.   |  
| `writeup.md`                 | Document summarizing the project 													|

[//]: # (Image References)

[image1]: ./example_images/car.jpg
[image2]: ./example_images/carHOG_BGR_1.jpg
[image3]: ./example_images/carHOG_BGR_2.jpg
[image4]: ./example_images/carHOG_BGR_3.jpg
[image5]: ./example_images/notcar.jpg
[image6]: ./example_images/notcarHOG_BGR_1.jpg
[image7]: ./example_images/notcarHOG_BGR_2.jpg
[image8]: ./example_images/notcarHOG_BGR_3.jpg
[image9]: ./example_images/car.jpg
[image10]: ./example_images/carHOG_HLS_1.jpg
[image11]: ./example_images/carHOG_HLS_2.jpg
[image12]: ./example_images/carHOG_HLS_3.jpg
[image13]: ./example_images/notcar.jpg
[image14]: ./example_images/notcarHOG_HLS_1.jpg
[image15]: ./example_images/notcarHOG_HLS_2.jpg
[image16]: ./example_images/notcarHOG_HLS_3.jpg
[image17]: ./example_images/car.jpg
[image18]: ./example_images/carHOG_YCrCb_1.jpg
[image19]: ./example_images/carHOG_YCrCb_2.jpg
[image20]: ./example_images/carHOG_YCrCb_3.jpg
[image21]: ./example_images/notcar.jpg
[image22]: ./example_images/notcarHOG_YCrCb_1.jpg
[image23]: ./example_images/notcarHOG_YCrCb_2.jpg
[image24]: ./example_images/notcarHOG_YCrCb_3.jpg
[image25]: ./example_images/car.jpg
[image26]: ./example_images/carHOG_HLS_orient_6.jpg
[image27]: ./example_images/carHOG_HLS_orient_9.jpg
[image28]: ./example_images/carHOG_HLS_orient_15.jpg
[image29]: ./example_images/car.jpg
[image30]: ./example_images/carHOG_HLS_ppc_4.jpg
[image31]: ./example_images/carHOG_HLS_ppc_8.jpg
[image32]: ./example_images/carHOG_HLS_ppc_16.jpg
[image33]: ./example_images/car.jpg
[image34]: ./example_images/carHOG_HLS_cpb_2.jpg
[image35]: ./example_images/carHOG_HLS_cpb_4.jpg
[image36]: ./example_images/carHOG_HLS_cpb_8.jpg
[image37]: ./example_images/test1_A_windows1.jpg
[image38]: ./example_images/test1_A_windows2.jpg
[image39]: ./example_images/test1_A_windows3.jpg

[image40]: ./output_images/test1_original.jpg
[image41]: ./output_images/test1_B_alldetectionsALL.jpg
[image42]: ./output_images/test1_C_heatmap.jpg
[image43]: ./output_images/test1_D_final.jpg

[image44]: ./output_images/test4_original.jpg
[image45]: ./output_images/test4_B_alldetectionsALL.jpg
[image46]: ./output_images/test4_C_heatmap.jpg
[image47]: ./output_images/test4_D_final.jpg

[image48]: ./example_images/18_original.jpg
[image49]: ./example_images/18_heatmap.jpg
[image50]: ./example_images/18_cumheatmap.jpg
[image51]: ./example_images/19_original.jpg
[image52]: ./example_images/19_heatmap.jpg
[image53]: ./example_images/19_cumheatmap.jpg
[image54]: ./example_images/20_original.jpg
[image55]: ./example_images/20_heatmap.jpg
[image56]: ./example_images/20_cumheatmap.jpg
[image57]: ./example_images/21_original.jpg
[image58]: ./example_images/21_heatmap.jpg
[image59]: ./example_images/21_cumheatmap.jpg
[image60]: ./example_images/22_original.jpg
[image61]: ./example_images/22_heatmap.jpg
[image62]: ./example_images/22_cumheatmap.jpg
[image63]: ./example_images/23_original.jpg
[image64]: ./example_images/23_heatmap.jpg
[image65]: ./example_images/23_cumheatmap.jpg
[image66]: ./example_images/24_original.jpg
[image67]: ./example_images/24_heatmap.jpg
[image68]: ./example_images/24_cumheatmap.jpg
[image69]: ./example_images/25_original.jpg
[image70]: ./example_images/25_heatmap.jpg
[image71]: ./example_images/25_cumheatmap.jpg

[video1]: ./project_video.mp4

####2. Histogram of Oriented Gradients (HOG)

#####a. Feature Extraction

HOG features measure the distribution of edge orientations across overlapping rectangular blocks in an image. HOG features have been found to work well as inputs to object classifiers. In this project, HOG features were extracted from all vehicle and non-vehicle images. I experimented with different parameters for computing the HOG feature descriptor:

######Color spaces:

The following examples show HOG features for various color spaces. The default BGR color-space is slightly more unreliable due to its variance under illumination conditions. I ended up using HOG features computed for all three channels of the HLS color space.

HOG features computed for all three channels of BGR color-space for an example car and non-car image:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]

![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

HOG features computed for all three channels of HLS color-space for an example car and non-car image:

![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]

HOG features computed for all three channels of YCrCb color-space for an example car and non-car image:

![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]

![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]

######Number of Orientation bins:

I experimented with 6, 9 and 15 number of orientation bins (2nd, 3rd and 4th images from left below). 6 was too few to capture all the relevant edge information, whereas there was not much difference in using 9 and 15. I used 9 because it meant using a more concise feature representation without degradation in classifier performance.

![alt text][image25]
![alt text][image26]
![alt text][image27]
![alt text][image28]

######Pixels per cell:

I experimented with (4, 4), (8, 8) and (16, 16) pixels per cells (2nd, 3rd and 4th images from left below). (8,8) pixels per cell struck the right balance in capturing relevant image attributes.

![alt text][image29]
![alt text][image30]
![alt text][image31]
![alt text][image32]

Cells per block:

I experimented with (2, 2), (4, 4) and (8, 8) cells per block bins (2nd, 3rd and 4th images from left below). This parameter did not have a meaningful impact on classifier performance on the test set, so I just used the default option (2,2).

![alt text][image33]
![alt text][image34]
![alt text][image35]
![alt text][image36]

On a random 50-50 split of the available training data, the parameters corresponding to the highest accuracy were: a) all three channels in the 'HLS' colorspace, b) 9 orientation bins, c) (8, 8) pixels per cell, and d) (2, 2) cells per block. The length of the HOG feature descriptor for each input image patch was 5292 (See `get_hog_features()` function in `train_classifier.py` for implementation details). Training a classifier on HOG features alone yielded a 98.5 percent classifier accuracy. However, when HOG feature vector was further augmented by computing spatial bin features of size (16, 16) and histogram of color features with 32 bins, the classifier accuracy on the held-out validation set increased to 99.02 percent. Thus, the total length of the feature vector for each input image (using HOG, spatial bin and color histogram features) was 6156.

#####b. SVM Training

The available training set of images (GTI_Far, GTI_Left, GTI_MiddleClose, GTI_Right and KITTI_extracted for Vehicles and Extras and GTI for Non-Vehicles) were read and the aforementioned feature vector was extracted for each (in the `extract_features()` function in `train_classifier.py`) and normalized using the StandardScaler. The entire set of images (8792 cars and 8968 non-cars) was then randomly divided into 50-50 train and validation split (to avoid overfitting). A LinearSVM was then trained on the training set (in `train_classifier.py`).

Note: I also experimented with training a more powerful classifier (An SVM with an 'rbf' kernel with the probability parameter set to True) in order to make more accurate predictions as well as to weight the predictions with the predicition probabilities (so that predictions with lower confidence can be filtered out), but both training the classifier and testing the vehicle detection pipeline took an infeasibly long time. 

####3. Sliding Window Search

#####a. Sliding Window Implementation and Parameters

A sliding window in 3 scales are used to search for vehicles in an image (See the `slide_window()` function in `detect_vehicles.py`). I experimented with different start and stop y-positions for the different scales based on the assumption that vehicles closest to the camera (i.e. the vehicle recording the images) appear larger and towards the bottom of the image frame, whereas vehicles far ahead of the image appear smaller and towards the middle of the image. The percent of overlap is also a higher value for the windows closer to the camera (i.e. larger windows at the bottom of the image) than at the top. The windows in the 3 scales are visualized ehre:

![alt text][image39]

![alt text][image38]

![alt text][image37]


For each of the windows, the pre-trained classifier is applied to make a prediction. Instead of computing HOG features for each window separately, a more efficient method was implemented, where the HOG feature was computed once for the entire image, and the descriptor for the window was computed by slicing the relevant portion of the HOG image (See the `find_cars()` function in `detect_vehicles.py`). If the classifier deems that the window contains a vehicle, the bounding-box/window gets added to the heatmap. In order to get clean bounding boxes from overlapping windows of different sizes as well as ignore spurious detections, a threshold is applied on the final heatmap (normalized by dividing the heatmap by the maximum number, so that the heatmap pixels range from 0 to 1) for each image. `scipy.ndimage.measurements.label()` was used to identify individual blobs in the thresholded heatmap. Bounding boxes were constructed to cover the area of each blob detected and assumed to correspond to a detected vehicle. 

#####b. Examples of Test Images on Pipeline

The following set of images show the different steps of the pipeline (See `test_images()` in `detect_vehicles.py` for implementation). The reliability of the classifier was improved mainly by: a) ignoring vehicle predictions with low confidence. This is implemented by thresholding the output of LinearSVC.decision_function() (which outputs the signed distance of the sample to the decision hyperplane) using a threshold of 0.5, and b) thresholding the normalized heatmap by a threshold of 0.5. This can be interpreted as pixels corresponding to detected vehicles have at least 50% of the maximum intensity of the heat map. Below are the steps of the pipeline on 2 test images:

Original images:

![alt text][image40]
![alt text][image44]

Images with predicted windows in multiple scales:

![alt text][image41]
![alt text][image45]

Heat maps representing the predicited windows:

![alt text][image42]
![alt text][image46]

Final bounding boxes obtained by thresholding the heatmap:

![alt text][image43]
![alt text][image47]

Results on the remaining test images can be found in the folder `output_images`.

####4. Video Implementation

#####a. Vehicle Detection in Videos

A VehicleDetector class was implemented (in `detect_vehicles.py`) to encapsulate the vehicle detection pipeline explained above and help keep track of heatmaps across multiple frames. The sliding-window search was used in every frame of the video to search for and identify vehicles. For each frame, the detected vehicles have been identified with a bounding box. Here's a [link to my video result](./project_video_out2.mp4).

Here is [another example](./project_video_out1.mp4) with a slight variation of the parameters (a lower heatmap threshold):

A high threshold successfully removes most false positives but also increases the amount of false negatives (i.e. the classifier fails to detect cars in some instances). Whereas decreasing the threshold removes most false negatives but increases the amount of false positives. In practice, outcomes corresponding to different settings of the threshold can be plotted in an ROC curve and the most suitable one be chosen for the application.

#####b. Filtering False Positives

At every frame, false positives are first minimized by only accepting predictions that are above some threshold of classifier confidence. The predictions are then further filtered out by computing a heatmap and thresholding it. For videos or image sequences, a cumulative heat map over n frames was used to further filter out false positives. This was based on the assumption that the vehicles have strong detections across consecutive frames but false positives do not. Below is an example over a sequence of 8 frames. The left images consist of the original frames, the middle images consist of independent heatmaps, whereas the right images consists of cumulative heatmaps normalized over the last 8 frames. You can see that false positives are more prominent in the independent heatmaps compared to the cumulative heatmaps, and therefore easily removed by thresholding. Finally, `scipy.ndimage.measurements.label()` was used to compute bounding boxes after identifying individual blobs in the thresholded cumulative heatmap.

![alt text][image48]
![alt text][image49]
![alt text][image50]

![alt text][image51]
![alt text][image52]
![alt text][image53]

![alt text][image54]
![alt text][image55]
![alt text][image56]

![alt text][image57]
![alt text][image58]
![alt text][image59]

![alt text][image60]
![alt text][image61]
![alt text][image62]

![alt text][image63]
![alt text][image64]
![alt text][image65]

![alt text][image66]
![alt text][image67]
![alt text][image68]

![alt text][image69]
![alt text][image70]
![alt text][image71]


####5. Discussion

The pipeline works fairly well but still struggles with some false positives (which can be dangerous in real-world driving scenarios). Training a more robust classifier (e.g. a CNN trained on a larger dataset of images) can alleviate this problem. Another drawback of the current approach is that although a simple Linear SVM is used for classification, it still does not run in real-time. Thus, optimizing the code (e.g. parallelizing the sliding window mechanism) can help make this approach feasible. Currently, the pipeline detects vehicles but does not track them. One extension would be to keep track of vehicles over frames, which can help eliminate the constant change in the vehicle bounding boxes.

