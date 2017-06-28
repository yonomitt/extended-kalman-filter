#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

float centerAngle(float phi) {
  // if phi is too big, bring it down 2pi at a time
  while (phi > M_PI) {
    phi -= 2 * M_PI;
  }
  
  // if phi is too small, bring it up 2pi at a time
  while (phi < -M_PI) {
    phi += 2 * M_PI;
  }
  
  // return the new phi that is between -pi and pi
  return phi;
}

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::UpdateCommon(const VectorXd &y) {
  // The common calculations for both the normal
  // and extended kalman filters
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Normal Kalman filter update
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Extended Kalman filter update
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];

  float ro = sqrt(px * px + py * py);
  float phi;
  float ro_dot;

  if (fabs(px) < 0.0001) {
    // p'x is too small, set phi to 0
    phi = 0.0;
  } else {
    phi = atan2(py, px);
  }

  if (fabs(ro) < 0.0001) {
    // ro is too small, so set ro' to 0
    ro_dot = 0.0;
  } else {
    ro_dot = (px * vx + py * vy) / ro;
  }

  VectorXd z_pred = VectorXd(3);
  z_pred << ro, phi, ro_dot;

  VectorXd y = z - z_pred;

  y[1] = centerAngle(y[1]);

  UpdateCommon(y);
}
