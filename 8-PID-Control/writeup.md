#**PID-Control-Project**

The goal of this project was to:

* Implement a PID controller that maneuvers the vehicle around the simulator track.
* Given the cross track error (CTE) and the velocity (mph), compute an appropriate steering angle by choosing good parameters for the PID controller.

####1. Reflection: Effect of the P,I,D components

The PID controller is a commonly used control mechanism that computes an appropriate value for the control variable by taking into account the difference between a reference point and a measured process variable [1](https://en.wikipedia.org/wiki/PID_controller). The value of the control variable is computed based on the coefficients for the following three terms:

#####P (Proportional): 

The proportional term defines an error proportional to the crosstrack error (CTE), which is a measure of the difference between the car's posiiton and the closest point on the reference trajectory.

When the coefficient for the proportional term is too low (< 0.01), the car does not adjust to the error quickly enough, whereas if the coefficient is set too high (> 0.33), the car adjusts too quickly to the error, frequently overshooting and leading to wild oscillations. A value in between (0.1) achieved a good balance.

#####D (Derivative):

The problem with the proportional controller is the oscillations caused by its tendency to over-compensate for the CTE. The derivative term in the PD controller eliminates the oscillations by smoothing the manner in which the error is reduced.

With the P controller coefficient set, the coefficient for the derivative term can be tuned. But if the D term is set too low (< 1.0), the car still oscillates, whereas if the D term is set too high (> 6.6), too much counter-steering is applied preventing the CTE from going down fast enough. Thus, a good intermediate value of (3.3) was chosen.

#####I (Integral):

In the presence of systematic bias (for example, in the presence of steering drift) the vehicle never reaches a small CTE for any setting of the PD parameters. Therefore, in order to minimize this bias, an integral term is added which takes it account the total CTE error over time. 

If the coefficient for the Integral term was set too high (> 0.01), the control system tries to over-compensate for the bias which leads to the car going off track. Although the car completes the lap with the I coefficient set to 0, the car seems to drive more on the center of the track when it is set to (0.001).

#####Video:

As described above, the coefficients for the controller was selected manually. The parameters were searched using multiples of 3 (..., 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, ...) Here is a [video](https://vimeo.com/219456430) of the car completing a lap for the final setting of the PID parameters.



