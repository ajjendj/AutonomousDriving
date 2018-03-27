Team Members:
-------------
Yu Wang - yuwangrpi@gmail.com [Team Lead]
Upul Bandara - upulbandara@gmail.com
Matt Alexander - mda54092@gmail.com
Ajjen Joshi - ajjendj@bu.edu
Rajan Chaudhry - rajanchaudhry@gmail.com

Nodes:
------

Waypoint Updater:
The waypoint updater node subscribes to the base_waypoints, current_pose and traffic_waypoints topics and uses the messages on these topics to publish a set of waypoints alongs with their target velocities to the final_waypoints topic. We do not subscribe to the obstacle_waypoints topic even though it is mentioned in the project overview.
The final_waypoints are calculated whenever a new vehicle position is obtained. If the vehicle position is the same as the previous one, then the message is ignored. If a new vehicle position is received, then the vehicle’s current position is used to determine the next set of waypoints in front of the vehicle. If a red light is present in front of the vehicle, then the vehicle is slowed down and stopped at a point which is 30 waypoints before the stop sign waypoint (30 was chosen in order to approximate the distance from light to the stop line). Once the light turns green, then waypoints are again calculated and the velocities are set based on what we received initially from the base_waypoints.

Twist Controller:
The twist controller subscribes to current_velocity, dbw_enabled and twist_cmd topics and publishes throttle, brake, and steering commands to the /vehicle/throttle_cmd, /vehicle/brake_cmd, and /vehicle/steering_cmd topics. The Yaw Controller provided by the project team is used to generate the steering angle based on linear and angular input velocity and target velocity. The PID controller is used to generate the throttle and brake values. If the throttle value returned by PID controller is negative, then throttle value is set to 0 and brake is applied based on the formula deceleration * self.vehicle_mass * self.wheel_radius. Deceleration is equal to -ve throttle. If the throttle value is positive then the brake value sent to the vehicle is 0. The PID input configuration parameters were chosen based on experimentation on the simulator and the best ones were used.

TL detector:
The traffic light detection node subscribes to base_waypoints, current_pose and image_color topics. During test We first find if there is a nearest upcoming traffic light. If there is, we find the waypoint nearest to the traffic light and send the image produced by the car's camera to the light detector for detection and color recognition. 

FRCNN Detector:
The traffic light detector is based on the FRCNN architecture. It was trained using the transfer learning technique. We started with `faster_rcnn_resnet101_coco` model downloaded from "Tensorflow detection model zoo" Github page and fine-tune it using Udacity's robag images. When fine-tuning is completed the model was exported as a `*.pd` file and saved inside the System- Integration project.

The exported model is used inside the `TLClassifier` (in `tl_classifier.py` file) class. The saved model is used to initialize the Tensorflow graph. The prediction actually happens inside the `get_classification` method. Images are inputted as `numpy` arrays. Then Tensorflow graph runs and returns the predicted class. Next, `get_classification` method decodes the output of the Tensorflow graph and returns the appropriate traffic light color indicator to the calling method (i.e. process_traffic_lights method of the `TLDetector` class.).
