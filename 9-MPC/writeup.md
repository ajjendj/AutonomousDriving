#**MPC-Control-Project**

The goal of this project was to:

* Implement a Model Predictive Control that maneuvers the vehicle around the simulator track.
* Compute the cross track error, as well as account for a 100 millisecond actuation latency.

####1. The Model

The state consists of four components: [x, y, psi, v] where x,y represnts the x,y position, psi represents the vehicle orientation and v represents the vehicle's velocity.

The actuator consists of two components: [delta, a] where delta represents the steering angle and a represents the acceleration (positive values corresponds to throttle, negative values correspond to brakes).

We also keep track of the cross track error and psi error as part of our state.

The update equations to update the state vectors consist of:
-> x_t+1 = x_t + v * (cos(psi)) * dt
-> y_t+1 = y_t + v * (sin(psi)) * dt
-> psi_t+1 = psi_t + (v/L_f)*delta*dt
-> v_t+1 = v_t + a * dt

####2. Timestep Length and Elapsed Duration (N and dt): 

N is the number of timesteps that the model accounts for, i.e. the higher the value of N, the further ahead the model has to predict. dt represents the length of a single timestep. The smaller the value of dt, the more frequently the model has to be evaluated. 

A setting of N = 15 and dt = 0.1 gave good results for the car to drive well around the lap at over 50mph. Short predictions for N < 10 are more responsive but not very accurate, especially for higher speeds. Large values of N (or very small values for dt), although more accurate, has a higher computational burden. 

####3. Polynomial Fitting and MPC Preprocessing

A polynomial of degree 3 was fitted and gave good results. Given that the the model is only predicting the trajectory for a short time into the future, a 3 degree polynomial is sufficient. A preprocessing step of transforming the waypoints into the vehicle coordinate system was performed.

####4. MPC with Latency

To account for a 100 millisecond actuation latency, the state update equations was used to update the vechicle model to predict the state after 100 milliseconds. This was then used as the initial state for the controller (although, the driving is slightly more unstable compared to a model without any latency).

#####Video:
Here is a [video](https://vimeo.com/221684893) of the car completing a lap for the final setting of the MPC parameters.



