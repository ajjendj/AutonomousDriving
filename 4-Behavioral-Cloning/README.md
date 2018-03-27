#**Behavioral Cloning** 

The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

####1. Submission includes all required files.

| File                         | Description                                                                        |
| ---------------------------- | ---------------------------------------------------------------------------------- |
| `model.py`                   | Creates and trains model using a generator that performs data augmentation on-the-fly.|
| `model.json`                 | JSON file containing model architecture.             								|
| `model.h5`                   | File with the trained model weights.                                               |
| `drive.py`                   | Sends real-time steering angles to the driving simulator after forward propagating the input image through the saved model. | 
| `writeup.md`                 | Document summarizing the project |




[//]: # (Image References)

[image1]: ./images/Model_diagram.png "Model Visualization"
[image2]: ./images/Data_unbalanced.png "Data Histogram"
[image3]: ./images/Data_balanced.png "Data Histogram after balancing"
[image4]: ./images/Data_raw.png "Example of raw image"
[image5]: ./images/Data_cropped.png "Example of image after cropping"
[image6]: ./images/Data_unflipped.png "Example of image before flipping"
[image7]: ./images/Data_flipped.png "Example of image after flipping"
[image8]: ./images/Data_center.png "Example of image from center camera"
[image9]: ./images/Data_left.png "Example of image from left camera"
[image10]: ./images/Data_right.png "Example of image from right camera"
[image11]: ./images/track1.gif "Clip of driving on track 1"
[image12]: ./images/track2.gif "Clip of driving on track 2"

---

####2. Submssion includes functional code.
After loading the track in the provided simulator, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable.

The `model.py` file contains the code for training the convolution neural network with the saved data, as well as saving the model. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

####4. Model Architecture

The model ('model.py' Lines 110-122) was inspired by the comma.ai model and consists of 3 convolutional layers, each with stride 2 (no pooling is performed): the first layer with 16 5x5 filters, the second layer with 32 5x5 filters, and the third layer with 64 3x3 filters. The output of the third convolutional layer is flattened and passed through 3 fully-connected layers of size 512, 100 and 20 that finally outputs a single floating point number representing the steering angle. All layers use RELU as the activation function. The data is normalized using a lambda layer and cropped using a Cropping2D layer. A visualization of the model can be found below:

![alt text][image1]

####5. Overfitting 

The model contains dropout layers after 2 of the fully connected layers in order to reduce overfitting. The initial training data was split into separate train and validation sets to ensure the model didn't overfit on the training data.

####6. Model parameters 

The model was trained using an adam optimizer, minimizing the mse error. The model was trained for 25 epochs when both train and validation errors decreased below 0.005.

####7. Training data

Although I recorded several laps driving through Track 1 using the simulator, I discovered that the beta simulator did not save images from the left and right cameras. Therefore, I ended up using the pre-recorded dataset that was provided. Because the recorded dataset consists mostly of the car driving on the center of the track, I used the images from the left and right cameras to help the model learn recovery behavior.

####8. Solution Design

I started with the [comma.ai steering model](https://github.com/commaai/research/blob/master/train_steering_model.py) and trained it on the raw dataset and discovered the car quickly went off track. These were the steps I took to build a model that was able to drive successfully around track 1 (and almost successfully around track 2)
* Remove dataset bias: As can be seen in the following figure, around half of the training data consisted of images with steering angle 0.0 (i.e. the car driving straight) which made it difficult for the model to learn to turn. Therefore, in my generator function I only added the images with 0 steering angles with a 10 percent probability. This helped balance out the data on which the model was trained.  

![alt text][image2]  

![alt text][image3]

* Crop input: The top 70 pixels (consisting of confusing scenery textures) and bottom 25 pixels (consisting of the car hood) of the image was cropped to help the model focus on the signal provided by the lanes. Here is an example of an image before and after cropping:  

![alt text][image4]  

![alt text][image5]

* Augment data: Each image with a non-zero steering was flipped and added to the training data. Here is an example of an image before and after flipping:

![alt text][image6]  

![alt text][image7]

* Add recovery data: In order for the model to learn to drive towards the center from a non-center location, images from the left and right cameras were added to the training set with their steering angles modified with an offset of +-0.2.

![alt text][image8]  

![alt text][image9]

![alt text][image10]

The train-validation split yielded 6429 raw images for training and 1608 for validation before augmentation and pre-processing. Because the augmentation is done inside the generator function, it is done on-the-fly.

Finally, I added a small modification to the throttle parameter in `drive.py` so that the throttle decreased proportionally to the steering angle (so that the speed decreased at turns). Although not required, this helps reduce the recovery move distances made by the car at the edge of the lanes, making the drive somewhat smoother.

Following these steps, the car was able to drive autonomously around the first track without leaving the road. On the second track, it fails to make one sharp left-turn, but otherwise reaches the end of the track autonomously.

####9. Results

The car drives around Track 1, although not as smoothly as I would like. This is probably due to the fact that the model was trained on only a fraction of images with steering angle 0, resulting on the model usually outputting a non-zero steering even on straight roads. The car also drives around Track 2, failing only on one location with a very sharp left turn. This shows that the model has generalized pretty well. The full video can be seen [here](https://vimeo.com/203756854).

![alt text][image11]

![alt text][image12]

Although a simple example, this project serves as a cool proof-of-concept of end-to-end learning for self-driving cars.