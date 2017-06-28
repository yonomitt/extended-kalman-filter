#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  //
  // Calculate the RMSE here.
  //

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // estimations should not be zero length
  if (estimations.size() == 0) {
    cout << "ERROR - no estimations passed in" << endl;
    return rmse;
  }

  // estimations and ground_truth should be the same length
  if (estimations.size() != ground_truth.size()) {
    cout << "ERROR - estimation and ground truth are different sizes" << endl;
    return rmse;
  }

  // accumulate squared residuals
  for(int i = 0; i < estimations.size(); ++i) {
    VectorXd diff = estimations[i] - ground_truth[i];
    VectorXd diffSq = diff.array() * diff.array();

    rmse += diffSq;
  }

  // calculate the mean
  rmse /= estimations.size();

  // calculate the square root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  //
  //  Calculate a Jacobian here.
  //

  MatrixXd Hj(3,4);

  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float distSq = px * px + py * py;

  // check division by zero (or close to zero)
  if (distSq < 0.0001) {
    cout << "CalculateJacobian() - Error - Division by Zero" << endl;
  } else {
    //compute the Jacobian matrix
    float dist = pow(distSq, 0.5);
    float dist_3_2 = pow(distSq, 1.5);

    Hj(0, 0) = px / dist;
    Hj(0, 1) = py / dist;
    Hj(0, 2) = 0.0;
    Hj(0, 3) = 0.0;

    Hj(1, 0) = -py / distSq;
    Hj(1, 1) = px / distSq;
    Hj(1, 2) = 0.0;
    Hj(1, 3) = 0.0;

    Hj(2, 0) = py * (vx * py - vy * px) / dist_3_2;
    Hj(2, 1) = px * (vy * px - vx * py) / dist_3_2;
    Hj(2, 2) = Hj(0, 0);
    Hj(2, 3) = Hj(0, 1);
  }

  // return the result
  return Hj;
}
