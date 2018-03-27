#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::cout;
using std::endl;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  ///* initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  ///* if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  ///* if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  ///* initial state vector
  x_ = VectorXd(5);

  ///* initial covariance matrix
  P_ = MatrixXd(5, 5);

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.0;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.50;

  ///* Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  ///* Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  ///* Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  ///* Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  ///* Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = n_x_ + 2;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  ///* previous timestamp
  previous_timestamp_ = 0;

  ///* Weights of sigma points
  weights_ = VectorXd::Zero(2 * n_aug_ + 1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i = 1; i < weights_.size(); i++) {
    weights_(i) = 0.5 / (n_aug_ + lambda_);
  }

  ///* sigma predicted matrix
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  ///* radar measurement dimension
  n_z_radar_ = 3;

  ///* lidar lidar dimension
  n_z_lidar_ = 2;

  ///* lidar measurement noise covariance matrix 
  R_lidar_ = MatrixXd::Zero(n_z_lidar_, n_z_lidar_);
  R_lidar_ << std_laspx_ * std_laspx_ , 0,
        0, std_laspy_ * std_laspy_;


  ///* radar measurement noise covariance matrix 
  R_radar_ = MatrixXd::Zero(n_z_radar_, n_z_radar_);
  R_radar_ << std_radr_ * std_radr_ , 0, 0,
          0, std_radphi_ * std_radphi_, 0,
          0, 0, std_radrd_ * std_radrd_;
}


UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_){

    float px = 0, py = 0, v = 0;

    //if sensor is radar
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Compute position and velocity from range (rho), bearing (phi) and radial velocity (rhodot)
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rhodot = meas_package.raw_measurements_[2];
      px = rho * cos(phi);
      py = rho * sin(phi);
      float vx = rhodot * cos(phi);
      float vy = rhodot * sin(phi);
      v = sqrt(vx * vx + vy * vy);

    //if sensor is lidar
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // Get position (LIDAR does not have velocity)
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];
    }

    // Initialize state and state covariance matrix
    x_ << px , py , v, 0, 0;
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 10, 0, 0,
          0, 0, 0, 10, 0,
          0, 0, 0, 0, 1;

    // Get first timestamp
    previous_timestamp_ = meas_package.timestamp_;

    // UKF is now initialized
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  previous_timestamp_ = meas_package.timestamp_;

  Prediction(dt);

  // print the output
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ 
    cout << "Sensor: RADAR" << endl;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER){ 
    cout << "Sensor: LIDAR" << endl;
  }

  cout << "x_Predict = " << endl << x_ << endl;
  cout << "----------" << endl;
  cout << "P_Predict = " << endl << P_ << endl;
  cout << "----------" << endl;

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
  else if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }

  // print the output
  cout << "x_Update = " << endl << x_ << endl;
  cout << "----------" << endl;
  cout << "P_Update = " << endl << P_ << endl;
  cout << "----------" << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

   /*****************************************************************************
   *  1. Create augmented mean and covariance
   ****************************************************************************/
  
  // Create augmented mean vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_*std_yawdd_;

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2 * n_aug_ + 1);

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

   /*****************************************************************************
   *  2. Predict sigma points
   ****************************************************************************/
  
  //predict sigma points
  for (int i = 0; i< 2*n_aug_+1; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * (sin (yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * (cos(yaw) - cos(yaw+yawd*delta_t));
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;
    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  /*****************************************************************************
   *  3. Predict mean and covariance
   ****************************************************************************/

  // predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  /*****************************************************************************
   *  1. Transform mean and covariance to measurement space
   ****************************************************************************/
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z_lidar_, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    
    // measurement model
    Zsig(0,i) = p_x; //px
    Zsig(1,i) = p_y; //py
  }

  NIS_laser_ = Update(meas_package, Zsig);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  /*****************************************************************************
   *  1. Transform mean and covariance to measurement space
   ****************************************************************************/
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z_radar_, 2 * n_aug_ + 1);
  
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
  }

  NIS_radar_ = Update(meas_package, Zsig);
}

/**
 * Common Update function
 * @param meas_package The measurement at k+1
 * @param Zsig Matrix of sigma points in measurement space
 */
double UKF::Update(MeasurementPackage meas_package, MatrixXd Zsig){

  // set variables according to sensor type
  int n_z;
  MatrixXd R;
  bool normalize = false;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ 
    n_z = n_z_radar_;
    R = R_radar_;
    normalize = true;
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
    n_z = n_z_lidar_;
    R = R_lidar_;
  }

  //mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i=0; i < 2*n_aug_+1; i++) {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    if (normalize) {
      //angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  S = S + R;

  /*****************************************************************************
   *  2. Compute cross-correlation between sigma points in state and measurment spaces
   ****************************************************************************/

  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    if (normalize) {
      //angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    }

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff = z - z_pred;

  if (normalize){ 
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  }

  /*****************************************************************************
   *  3. Update state mean and covariance matrix
   ****************************************************************************/

  // Update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // Compute Normalized Innovation Squared measure for lidar measurements
  double NIS = z_diff.transpose() * S.inverse() * z_diff;
  return NIS;

}