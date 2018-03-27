#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {

	// Initialize Errors
  	p_error = 0;
  	i_error = 0;
  	d_error = 0;

	// Initialize Coefficients
	this->Kp = Kp;
  	this->Ki = Ki;
  	this->Kd = Kd;
}

void PID::UpdateError(double cte) {

	// Update each component of error
	double prev_cte = p_error;
	p_error = cte;
	d_error = cte - prev_cte;
	i_error += cte;
	
}

double PID::TotalError() {

	// Compute total error as sum of all error components
	return - (Kp * p_error + Ki * i_error + Kd * d_error);
}

