from yaw_controller import YawController
from pid import PID
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel,
                 max_steer_angle, decel_limit, accel_limit, brake_deadband,
                 vehicle_mass, wheel_radius, *args, **kwargs):
        self.yaw_controller = YawController(wheel_base, steer_ratio, min_speed,
                                            max_lat_accel, max_steer_angle)
        self.pid = PID(1.5, 0.1, 0.1, mn=decel_limit, mx=accel_limit)
        self.brake_deadband = brake_deadband
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius

    def control(self, linear_velocity, angular_velocity, current_velocity,
                time_diff, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        steer = self.yaw_controller.get_steering(linear_velocity, angular_velocity, current_velocity)

        velocity_diff = linear_velocity - current_velocity
        acceleration = self.pid.step(velocity_diff, time_diff)
        throttle = acceleration
        brake = 0
        if acceleration < 0:
            deceleration = -acceleration
            throttle = 0
            if deceleration < self.brake_deadband:
                deceleration = 0.0
            brake = deceleration * self.vehicle_mass * self.wheel_radius

        #if throttle > 0.4:
        #    throttle = 0.4
        #rospy.log('using throttle, brake: {}, {}'.format(throttle, brake))

        return throttle, brake, steer
