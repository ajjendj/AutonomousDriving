#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from copy import deepcopy
import math
from tf.transformations import euler_from_quaternion

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.
As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.
Once you have created dbw_node, you will update this node to use the status of traffic lights too.
Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
MAX_DECEL = 1.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.base_waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.base_waypoints_bkup = None
        self.pose = None

        self.prev_pose = None
        self.prev_next_waypoints = None
        self.traffic_light_index = -1

        rospy.spin()

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg.pose
        if self.pose is not None and self.base_waypoints is not None:
            if(self.prev_pose == msg.pose):
                rospy.loginfo("Same pose recieved.")
                #lane = Lane()
                #lane.waypoints = self.prev_next_waypoints
                #self.final_waypoints_pub.publish(lane)
            else:
                self.loop()

        self.prev_pose = msg.pose

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints.waypoints
        self.base_waypoints_bkup = deepcopy(waypoints.waypoints)

        rospy.logwarn("Got base waypoints and copy")

        self.base_waypoints_sub.unregister()

    def loop(self):
        # get vehicle pose
        car_x = self.pose.position.x
        car_y = self.pose.position.y
        car_z = self.pose.position.z
        car_o = self.pose.orientation
        #rospy.logwarn('car x, y: {}, {}'.format(car_x, car_y))
        car_q = (car_o.x, car_o.y, car_o.z, car_o.w)
        car_roll, car_pitch, car_yaw = euler_from_quaternion(car_q)

        # compute list of waypoints in front of the car
        waypoints_ahead = []
        closest_waypoint = -1
        stop_before_redlight_index = self.traffic_light_index - 5
        #rospy.logwarn("Stop index {}".format(stop_before_redlight_index))
        # for each waypoint w
        for w in range(len(self.base_waypoints)):
            wp = self.base_waypoints[w]
            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y
            wp_z = wp.pose.pose.position.z
            # check if waypoint is in front of the car
            wp_ahead = ((wp_x - car_x) * math.cos(car_yaw) +
                        (wp_y - car_y) * math.sin(car_yaw)) > 3.0
            # ignore if waypoint is not
            if not wp_ahead:
                continue

            if closest_waypoint == -1:
                closest_waypoint = w
                #rospy.logwarn("Closest waypoint is {}".format(w))
            wp.twist.twist.linear.x = 0.0

            if stop_before_redlight_index > 0 and w > stop_before_redlight_index:
                continue

            # calculate distance between waypoint and car and store (waypoint, distance) tuple
            wp_dist = math.sqrt((car_x - wp_x) ** 2 + (car_y - wp_y) ** 2 +(car_z - wp_z)**2)

            #Setting velocity to 10.
            wp.twist.twist.linear.x = 4.47#self.base_waypoints_bkup[w].twist.twist.linear.x

            waypoints_ahead.append((wp, wp_dist))

        # sort waypoints by distance
        waypoints_ahead = sorted(waypoints_ahead, key=lambda x: x[1])[:LOOKAHEAD_WPS]

        # get list of waypoints (from sorted list of (waypoint, distance) tuples)
        wps_ahead = [wp[0] for wp in waypoints_ahead]

        if closest_waypoint < stop_before_redlight_index:
            #rospy.logwarn("Two variables: {} < {} : {}".format(closest_waypoint, self.traffic_light_index, len(wps_ahead)))
            wps_ahead = self.decelerate(wps_ahead)

        lane = Lane()
        #self.prev_next_waypoints = wps_ahead
        lane.waypoints = wps_ahead

        # publish lane to final waypoints
        self.final_waypoints_pub.publish(lane)

    def decelerate(self, waypoints):
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.distance1(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
        return waypoints

    def distance1(self, p1, p2):
        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
        return math.sqrt(x*x + y*y + z*z)

    def traffic_cb(self, msg):
        do_loop = True
        if(self.traffic_light_index == msg.data):
            do_loop = False

        self.traffic_light_index = msg.data
        #rospy.logwarn("Got traffic light ----- {}".format(self.traffic_light_index))
        if do_loop:
            self.loop()
        #if msg.data > -1:
        #    rospy.logwarn("Red light ahead")
        #else:
        #    rospy.logwarn("Green light")

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def _print_selected_waypoints(self, waypoints):
        for w in range(len(waypoints)):
            wp = waypoints[w]
            wp_x = wp.pose.pose.position.x
            wp_y = wp.pose.pose.position.y
            wp_z = wp.pose.pose.position.z
            rospy.loginfo('point: {} {} {}'.format(wp_x, wp_y, wp_z))
        rospy.loginfo('=========================================')


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')