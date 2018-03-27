#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2( (map_y-y),(map_x-x) );

	double angle = abs(theta-heading);

	if(angle > pi()/4)
	{
		closestWaypoint++;
	}

	return closestWaypoint;

}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

// Gets current lane (0: left lane, 1: middle lane, 2: right lane)
int getCurrentLane(double car_d)
{
	int current_lane = 0;
	if ((car_d <= 8) && (car_d > 4))
	{
		current_lane = 1;
	}
	else if ((car_d <= 4) && (car_d > 0))
	{
		current_lane = 0;
	}
	else if ((car_d <= 12) && (car_d > 0))
	{
		current_lane = 2;
	}
	return current_lane;
}

// Check if any car in front of our vehicle is too close (for comfort)
bool checkTooClose(vector<vector<double>> sensor_fusion, int current_lane, double car_s, int prev_size, int distance_threshold)
{	
	bool too_close = false;
	for (int i = 0; i < sensor_fusion.size(); i++)
	{
		float d = sensor_fusion[i][6];
		if (d < (2+4*current_lane + 2) && d > (2 + 4*current_lane - 2))
		{
			double vx = sensor_fusion[i][3];
			double vy = sensor_fusion[i][4];
			double check_speed = sqrt(vx*vx + vy*vy);
			double check_car_s = sensor_fusion[i][5];

			check_car_s += ((double)prev_size*.02*check_speed);

			if ((check_car_s > car_s) && ((check_car_s-car_s)<distance_threshold))
			{
				too_close = true; 
			}
		}
	}
	return too_close;
}

// Check if lane-change from current_lane to target_lane is safe
// Lane-change is deemed to be safe if:
// a) Distance to closest car in front of our vehicle in the target lane is greater than front_distance_threshold
// b) Distance to closest car behind our vehicle in the target lane is greater than back_distance_threshold
bool isLaneChangeSafe(vector<vector<double>> sensor_fusion, int current_lane, int target_lane, double car_s, 
					  int prev_size, int front_distance_threshold, int back_distance_threshold)
{
	bool is_safe = true;
	std::cout << "Exploring lane change: (" << current_lane << "->" << target_lane << ")" << std::endl;
	bool car_infront = false;
	bool car_behind = false;

	for (int i = 0; i < sensor_fusion.size(); i++)
	{
		float d = sensor_fusion[i][6];
		if (d < (2+4*target_lane+2) && d > (2 + 4*target_lane-2))
		{
			double vx = sensor_fusion[i][3];
			double vy = sensor_fusion[i][4];
			double check_speed = sqrt(vx*vx + vy*vy);
			double check_car_s = sensor_fusion[i][5];

			check_car_s += ((double)prev_size*.02*check_speed);
			if ((check_car_s > car_s) && ((check_car_s-car_s)<front_distance_threshold)) //if car ahead of me
			{
				car_infront = true;
				is_safe = false;
			}
			if ((check_car_s < car_s) && (car_s-check_car_s)<back_distance_threshold) //if car behind me
			{
				car_behind = true;
				is_safe = false;
			}

		}
	}
	return is_safe;

}

// Gets target lane by checking if lane-change is feasible
// If car is in lanes 0 or 2, we check if changing to lane 1 is feasible
// If car is in lane 1, we first check if changing to the left lane (0) is feasible
// If changing to the left lane is not safe, we check if changing to the right lane is feasible
// If lane-change is not feasible we stay in the current lane
int getTargetLane(vector<vector<double>> sensor_fusion, int current_lane, double car_s, 
				  int prev_size, int front_distance_threshold, int back_distance_threshold)
{
	int target_lane = current_lane;
	if (current_lane == 1)
	{
		bool is_safe = isLaneChangeSafe(sensor_fusion, current_lane, current_lane - 1, car_s, prev_size, front_distance_threshold, back_distance_threshold);
		if (is_safe) 
		{	
			std::cout << "Changing lanes (1 -> 0)" << std::endl;
			target_lane = 0;
			
		} 
		else //car in lane 1 is too close, so don't change lanes
		{
			std::cout << "Not safe for lane change (1 -> 0)" << std::endl;

			bool is_safe = isLaneChangeSafe(sensor_fusion, current_lane, current_lane + 1, car_s, prev_size, front_distance_threshold, back_distance_threshold);
			if (is_safe) 
			{	
				std::cout << "Changing lanes (1 -> 2)" << std::endl;
				target_lane = 2;
				
			} 
			else //car in lane 1 is too close, so don't change lanes
			{
				std::cout << "Not safe for lane change (1 -> 2)" << std::endl;
				target_lane = 1;
				
			}
			
		}
	}

	else if (current_lane == 0)
	{
		bool is_safe = isLaneChangeSafe(sensor_fusion, current_lane, current_lane + 1, car_s, prev_size, front_distance_threshold, back_distance_threshold);
		if (is_safe) 
		{
			std::cout << "Changing lanes (0 -> 1)" << std::endl;
			target_lane = 1;	
		} 
		else
		{
			std::cout << "Not safe for lane change (0 -> 1)" << std::endl;
			target_lane = 0;
		}
	}

	else if (current_lane == 2)
	{
		bool is_safe = isLaneChangeSafe(sensor_fusion, current_lane, current_lane - 1, car_s, prev_size, front_distance_threshold, back_distance_threshold);
		if (is_safe) 
		{
			std::cout << "Changing lanes (2 -> 1)" << std::endl;
			target_lane = 1;	
		} 
		else
		{
			std::cout << "Not safe for lane change (2 -> 1)" << std::endl;
			target_lane = 2;
		}
	}
	return target_lane;
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  double ref_vel = 0;
  int lane = 1;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

    h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy, &ref_vel, &lane](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
   
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];
            
            // Get size of previous path     
			int prev_size = previous_path_x.size();

			if (prev_size > 0)
			{
				car_s = end_path_s;
			}
			
			// Get current lane of car
			int current_lane = getCurrentLane(car_d);
			int target_lane = current_lane;

			// Check if the car in front of our vehicle is too close
			// currently the distance threshold is set to 30m
			bool too_close = checkTooClose(sensor_fusion, current_lane, car_s, prev_size, 30);

			// If car in front of our vehicle is too close
			if (too_close)
			{
				// Get target lane (which is different from current lane if lane-change is feasible)
				target_lane = getTargetLane(sensor_fusion, current_lane, car_s, prev_size, 35, 15);
				// Slow down
				ref_vel -= .224;
			}
			else if (ref_vel < 49.5)
			{
				// Speed up if velocity is below speed limit
				ref_vel += .224;
			}

			//Following code mostly from project walk-through
			
			// Create a list of widely spaced (x,y) waypoints 
			// The waypoints will interpolated with a spline
			vector<double> ptsx;
			vector<double> ptsy;
			
			// Reference x,y and yaw states
			double ref_x = car_x;
			double ref_y = car_y;
			double ref_yaw = deg2rad(car_yaw);
			
			// if previous size is almost empty, use the car as starting reference
			if (prev_size < 2)
			{
				// use the two points that make the path tangent to the car
				double prev_car_x = car_x - cos(car_yaw);
				double prev_car_y = car_y - sin(car_yaw);
				
				ptsx.push_back(prev_car_x);
				ptsx.push_back(car_x);
				
				ptsy.push_back(prev_car_y);
				ptsy.push_back(car_y);
			}
			// else use the previous path's end point as starting reference
			else {
				ref_x = previous_path_x[prev_size-1];
				ref_y = previous_path_y[prev_size-1];
				
				double ref_x_prev = previous_path_x[prev_size-2];
				double ref_y_prev = previous_path_y[prev_size-2];
				ref_yaw = atan2(ref_y-ref_y_prev, ref_x-ref_x_prev);			

				ptsx.push_back(ref_x_prev);
				ptsx.push_back(ref_x);

				ptsy.push_back(ref_y_prev);
				ptsy.push_back(ref_y);
			}

			// create waypoints 30, 60 and 90 meters ahaed of current position
			vector<double> next_wp0 = getXY(car_s+30,(2+4*target_lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
			vector<double> next_wp1 = getXY(car_s+60,(2+4*target_lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
			vector<double> next_wp2 = getXY(car_s+90,(2+4*target_lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

			ptsx.push_back(next_wp0[0]);
			ptsx.push_back(next_wp1[0]);
			ptsx.push_back(next_wp2[0]);

			ptsy.push_back(next_wp0[1]);
			ptsy.push_back(next_wp1[1]);
			ptsy.push_back(next_wp2[1]);

			// shift to local coordinate system of the car
			for (int i = 0; i < ptsx.size(); i++)
			{
				double shift_x = ptsx[i] - ref_x;
				double shift_y = ptsy[i] - ref_y;

				ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y*sin(0 - ref_yaw)); 
				ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y*cos(0 - ref_yaw));
			}	

			// fit spline
			tk::spline s;
			s.set_points(ptsx, ptsy);

			// create trajectory of 50 x,y values
           	vector<double> next_x_vals;
          	vector<double> next_y_vals;
          	
          	// use values from previous paths so that the trajectories change smoothly
           	for(int i = 0; i < previous_path_x.size(); i++)
            {
				next_x_vals.push_back(previous_path_x[i]);
 	            next_y_vals.push_back(previous_path_y[i]);
            }

            // break of spline points at (x,y) points such that reference velocity is maintained
    		//   and fill up the remainder of the trajectory waypoints
    		//   remember to apply a transformation to shift back to global coordinates
			double target_x = 30.0;
			double target_y = s(target_x);
			double target_dist = sqrt((target_x)*(target_x)+(target_y)*(target_y));

			double x_add_on = 0;

			for (int i = 1; i <= 50 - previous_path_x.size(); i++)
			{
				double N = (target_dist/(.02*ref_vel/2.24));
				double x_point = x_add_on+(target_x)/N;
				double y_point = s(x_point);

				x_add_on = x_point;
				
				double x_ref = x_point;
				double y_ref = y_point;
				
				x_point = (x_ref * cos(ref_yaw) - y_ref*sin(ref_yaw));
				y_point = (x_ref * sin(ref_yaw) + y_ref*cos(ref_yaw));

				x_point += ref_x;
				y_point += ref_y;

				next_x_vals.push_back(x_point);
				next_y_vals.push_back(y_point);
			}

	        json msgJson;
			msgJson["next_x"] = next_x_vals;
	      	msgJson["next_y"] = next_y_vals;

	      	auto msg = "42[\"control\","+ msgJson.dump()+"]";

	      	//this_thread::sleep_for(chrono::milliseconds(1000));
	      	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
















































































