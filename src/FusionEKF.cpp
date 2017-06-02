#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

float centerAngle(float phi) {
  // if phi is too big, bring it down 2pi at a time
  while (phi > M_PI) {
    phi -= 2 * M_PI;
  }
  
  // if phi is too small, bring it up 2pi at a time
  while (phi <= -M_PI) {
    phi += 2 * M_PI;
  }

  // return the new phi that is between -pi and pi
  return phi;
}

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225,      0,
                   0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09,      0,    0,
                 0, 0.0009,    0,
                 0,      0, 0.09;

  // measurement matrix - laser
  H_laser_ << 1, 0, 0, 0,
	      0, 1, 1, 1;

  //
  // Finish initializing the FusionEKF.
  // Set the process and measurement noises
  //

  // create a 4D state vector
  ekf_.x_ = VectorXd(4);

  // state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0,    0,    0,
             0, 1,    0,    0,
             0, 0, 1000,    0,
             0, 0,    0, 1000;
  
  // create the process noise covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);

  // the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
             0, 1, 0, 1,
             0, 0, 1, 0,
             0, 0, 0, 1;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    //
    // Initialize the state ekf_.x_ with the first measurement.
    // Create the covariance matrix.
    // Remember: you'll need to convert radar from polar to cartesian coordinates.
    //

    previous_timestamp_ = measurement_pack.timestamp_;

    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      float ro = measurement_pack.raw_measurements_[0];
      float phi = measurement_pack.raw_measurements_[1];
      float ro_prime = measurement_pack.raw_measurements_[2];

      // center phi betwen -pi and pi
      phi = centerAngle(phi);

      ekf_.x_[0] = ro * cos(phi);
      ekf_.x_[1] = ro * sin(phi);
      ekf_.x_[2] = ro_prime * cos(phi);
      ekf_.x_[3] = ro_prime * sin(phi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      ekf_.x_[0] = measurement_pack.raw_measurements_[0];
      ekf_.x_[1] = measurement_pack.raw_measurements_[1];
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // dt needs to be in seconds, but timestamp is in microseconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // 1. Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // 2. Set the process noise covariance matrix Q
  float noise_ax = 9;
  float noise_ay = 9;

  float dt2 = dt * dt;
  float dt3 = dt2 * dt;
  float dt4 = dt2 * dt2;

  float dt4_4 = dt4 / 4;
  float dt3_2 = dt3 / 2;

  ekf_.Q_ << dt4_4 * noise_ax,                0, dt3_2 * noise_ax,                0,
                            0, dt4_4 * noise_ay,                0, dt3_2 * noise_ay,
             dt3_2 * noise_ax,                0,   dt2 * noise_ax,                0,
                            0, dt3_2 * noise_ay,                0,   dt2 * noise_ay;

  // 3. Call the Kalman Filter predict() function
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
