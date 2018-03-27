#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml

import numpy as np

STATE_COUNT_THRESHOLD = 3

LIGHT_STATES = {0: 'red', 1: 'yellow', 2: 'green', 4: 'unknown'}

USE_DETECTOR = False

DO_PROCESS_TRAFFIC_LIGHT = True

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

	self.image = None

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, lane):
	self.waypoints = lane.waypoints
	rospy.loginfo('waypoints: {}'.format(len(self.waypoints)))

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

	self.camera_image.encoding = "rgb8"
	self.image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")


	if DO_PROCESS_TRAFFIC_LIGHT:
            light_wp, state = self.process_traffic_lights()
	else:
	    light_wp = -1
	    state = TrafficLight.UNKNOWN

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp

	    rospy.loginfo('publish light waypoint {}'.format(light_wp))

            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
	car_position = None

	if (self.waypoints):
	    
	    # get the closest waypoint
	    wpts = np.zeros((len(self.waypoints),2), dtype=np.float32)

	    for i, wpt in enumerate(self.waypoints):
		wpts[i,0] = wpt.pose.pose.position.x
		wpts[i,1] = wpt.pose.pose.position.y

	    dists = (wpts[:,0] - pose.position.x) ** 2 + (wpts[:,1] - pose.position.y) ** 2

	    car_position = np.argmin(dists)
	return car_position

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #TODO Use tranform and rotation to calculate 2D position of light in image

        x = 0
        y = 0

        return (x, y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_positions = self.config['light_positions']   # traffic light's stop line positions
	print light_positions[0][0], light_positions[0][1], light_positions[0]

        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)
	
	rospy.loginfo('car_position: {}'.format(car_position))

        #TODO find the closest visible traffic light (if one exists)
	if(self.waypoints):
	    waypoints = self.waypoints[car_position:] + self.waypoints[:car_position]

	    # get the closest traffic light in front of car
	    light_idx = None
	    light_wp = None

	    # check 100 way points ahead
	    for i in range(1, 100):
		dists = []

		"""for light in self.lights:
		    # ideally this distance should be geodesic instead of euclidean 
		    dx = light.pose.pose.position.x - waypoints[i].pose.pose.position.x
		    dy = light.pose.pose.position.y - waypoints[i].pose.pose.position.y
		    dist = np.sqrt(dx**2 + dy**2)
		    dists.append(dist) """

		for light in light_positions:
		    dx = light[0] - waypoints[i].pose.pose.position.x
		    dy = light[1] - waypoints[i].pose.pose.position.y
		    dist = np.sqrt(dx**2 + dy**2)
		    dists.append(dist)

		light_idx = np.argmin(dists)

		# check if the distance is less than 100 meters 
		if dists[light_idx] < 100:
		   # rospy.loginfo('{}th light is close to the car: {}'.format(light_idx, dists[light_idx]))
		   break
                else:
		   light_idx = None

	    if light_idx is not None:
		# find index of waypoint closes to the upcoming traffic light

		if 0:
		   light_wp = self.get_closest_waypoint(self.lights[light_idx].pose.pose)
		else:
		   # use traffic light's stop line positions
		   # light_wp1 = self.get_closest_waypoint(self.lights[light_idx].pose.pose) 

		   light_pose = self.pose.pose
		   light_pose.position.x = light_positions[light_idx][0]
		   light_pose.position.y = light_positions[light_idx][1]

		   light_wp = self.get_closest_waypoint(light_pose)

		   # print self.lights[light_idx].pose.pose.position, light_positions[light_idx], waypoints[light_wp].pose.pose.position
		   # print light_wp1, light_wp

		# print light_wp, car_position

	 	if light_wp >= car_position:
		   # check if the car already passed the traffic light
		   # this is just a temporary workaround since 
		   # we can't use image to decide if the light is visible in the camera image or not

		   if not USE_DETECTOR:
		   	# get the light state directly from the simulator
		   	state = self.lights[light_idx].state
		   else:
			# apply detector and classifier
			state = self.get_light_state(light)
		
	    	   rospy.loginfo('the light is of state {}'.format(LIGHT_STATES[state]))
		   rospy.loginfo('waypoint closes to the upcoming traffic light {}'.format(light_wp))

		   return light_wp, state
		
	        # cv2.imshow('image', self.image)
		# cv2.waitKey(1)

	   
        # if light:
        #    state = self.get_light_state(light)
        #     return light_wp, state
        # self.waypoints = None

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
