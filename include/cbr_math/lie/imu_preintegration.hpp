// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_MATH__LIE__IMU_PREINTEGRATION_HPP_
#define CBR_MATH__LIE__IMU_PREINTEGRATION_HPP_

#include <sophus/se3.hpp>

#include <chrono>
#include <tuple>
#include <utility>
#include <vector>

#include "Tn.hpp"
#include "group_product.hpp"

namespace bps::preint
{
template<typename T>
using State = cbr::lie::GroupProduct<
  T, 0,
  Sophus::SE3,  // pose
  cbr::lie::T3  // linear velocity (vx, vy, vz)
>;

static constexpr std::size_t poseIdx = 0;
static constexpr std::size_t velIdx = 1;
static constexpr std::size_t DoF = State<double>::DoF;

template<typename T>
using Derivative = typename State<T>::Tangent;

template<typename T>
using Pose = Sophus::SE3<T>;

template<typename T>
using Rotation = Sophus::SO3<T>;

template<typename T>
using Position = Sophus::Vector3<T>;

template<typename T>
using Velocity = cbr::lie::T3<T>;

template<typename T>
using Covariance = Eigen::Matrix<T, DoF, DoF>;
using Covarianced = Covariance<double>;

using Time = std::chrono::nanoseconds;
using Accel = Sophus::Vector3d;
using Gyro = Sophus::Vector3d;
static constexpr std::size_t accelSize = Accel::RowsAtCompileTime;
static constexpr std::size_t gyroSize = Gyro::RowsAtCompileTime;
static constexpr std::size_t measSize = accelSize + gyroSize;

template<typename T>
using AccelBias = Eigen::Matrix<T, accelSize, 1>;
template<typename T>
using GyroBias = Eigen::Matrix<T, gyroSize, 1>;
static const AccelBias<double> zeroAccelBias = AccelBias<double>::Zero();
static const GyroBias<double> zeroGyroBias = GyroBias<double>::Zero();
using AccelCov = Eigen::Matrix<double, accelSize, accelSize>;
using GyroCov = Eigen::Matrix<double, gyroSize, gyroSize>;

// Relative pose should be in SE_2_3
using Meas = std::tuple<Time, Accel, Gyro>;
template<typename T>
using MeasBiased = std::tuple<Time, Accel, Gyro, AccelBias<T>, GyroBias<T>>;
static constexpr std::size_t timeIdx = 0;
static constexpr std::size_t accelIdx = 1;
static constexpr std::size_t gyroIdx = 2;
static constexpr std::size_t accelBiasIdx = 3;
static constexpr std::size_t gyroBiasIdx = 4;

using MeasCov = Eigen::Matrix<double, measSize, measSize>;
using A = Eigen::Matrix<double, DoF, DoF>;
using B = Eigen::Matrix<double, DoF, measSize>;

template<typename T>
using Container = std::vector<T>;

template<typename T>
using Kinematics =
  std::function<void (const State<T> & state, Derivative<T> & deriv, const double)>;

/* -------------------------------------------------------------------------- */
/*                             IMU Lie Integration                            */
/* -------------------------------------------------------------------------- */
// This approach computes the relative pose with a templated constant bias.

template<typename T = double>
Derivative<T> imu_kinematics(
  const State<T> & state, const Accel & accel, const Gyro & gyro,
  const AccelBias<T> & accelBias, const GyroBias<T> & gyroBias
)
{
  AccelBias<T> accelNet = accel - accelBias;
  GyroBias<T> gyroNet = gyro - gyroBias;
  return (Eigen::Matrix<T, DoF, 1>() <<
         std::get<velIdx>(state).translation(), gyroNet, accelNet
  ).finished();
}

template<typename T = double>
Derivative<T> imu_kinematics(const State<T> & state, const MeasBiased<T> & meas)
{
  return imu_kinematics(
    state, std::get<accelIdx>(meas), std::get<gyroIdx>(meas),
    std::get<accelBiasIdx>(meas), std::get<gyroBiasIdx>(meas)
  );
}

template<typename T = double>
Covariance<T> imu_covariance_propagation(
  Covariance<T> & covariance, const A & a, const B & b, const AccelCov & accelCov,
  const GyroCov & gyroCov)
{
  MeasCov measCov = MeasCov::Zero();
  measCov.block(0, 0, accelSize, accelSize) = accelCov;
  measCov.block(accelSize, accelSize, gyroSize, gyroSize) = gyroCov;

  return a * covariance * a.transpose() + b * measCov * b.transpose();
}

/**
 * @brief Computes the static convariance associated with an IMU measurement sequence and const bias
 * \del T_{k+1} = Ad_{Exp(-v)}\del T_{k} + A \delT_{k} + B \nu
 * @param measContainer Sequence of measurements
 * @param accelBias Assumend accelerometer measurement bias
 * @param gyroBias Assumed gyrosocope measurement bias
 * @param accelCov Accelometer variances
 * @param gyroCov Gyrosocpe variances
 * @return Covarianced Resulting covariance matrix of the relative pose
 */
template<typename T = double>
Covarianced propagate_imu_covariance(
  const Container<Meas> & measContainer, const AccelBias<T> & accelBias,
  const GyroBias<T> & gyroBias, const AccelCov & accelCov, const GyroCov & gyroCov
)
{
  A a = A::Zero();
  a.block(0, 6, 3, 3) = Eigen::Matrix3d::Identity();
  B b = B::Zero();
  b.block(3, 3, 3, 3) = Eigen::Matrix3d::Identity();
  b.block(6, 0, 3, 3) = Eigen::Matrix3d::Identity();

  State<T> state{};
  Covarianced covariance = Covarianced::Zero();
  for (auto it = measContainer.begin(); it != measContainer.end() - 1; ++it) {
    const T dt = std::chrono::duration_cast<std::chrono::duration<T>>(
      std::get<timeIdx>(*(it + 1)) - std::get<timeIdx>(*it)
      ).count();
    typename State<T>::Tangent vdt =
      imu_kinematics(state, std::get<accelIdx>(*it), std::get<gyroIdx>(*it), accelBias, gyroBias) *
      dt;
    State<T> dX = State<T>::exp(vdt);
    // TODO(jbm): proper covariance impl with J_r or higher-order BCH terms
    covariance = imu_covariance_propagation(
      covariance,
      // dX.inverse().Adj() + dt * a, b * dt, accelCov, gyroCov);
      dX.inverse().Adj() + a, b, accelCov, gyroCov
    );
    state = state * dX;
  }
  return covariance;
}

/**
 * @brief Computes relative pose with constant biases over the sequence of IMU measurements
 * Relative pose is computed between the timestamps of the first and the last measurements
 * @tparam T Scalar type for state and biases
 * @param measContainer Sequence of IMU measurements
 * @param accelBias<T> Accelerometer bias for sequence interval
 * @param gyroBias<T> Gyroscope bias for sequence interval
 * @return State<T> Resulting relative pose
 */
template<typename T = double>
State<T> integrate_imu_measurements(
  const Container<Meas> & measContainer, const AccelBias<T> & accelBias,
  const GyroBias<T> & gyroBias)
{
  State<T> finalState{};
  for (auto it = measContainer.begin(); it != measContainer.end() - 1; ++it) {
    const double dt = std::chrono::duration_cast<std::chrono::duration<double>>(
      std::get<timeIdx>(*(it + 1)) - std::get<timeIdx>(*it)
      ).count();
    finalState = finalState * State<T>::exp(
      dt * imu_kinematics(
        finalState, std::get<accelIdx>(*it), std::get<gyroIdx>(*it), accelBias, gyroBias
      )
    );
  }
  return finalState;
}

/* -------------------------------------------------------------------------- */
/*                 Incremental IMU On-Manifold Preintegration                 */
/* -------------------------------------------------------------------------- */
// This approach updates the pose everytime a measurement is added, with a zero-overhead retrieval
template<typename T = double>
class ImuPreintegrator
{
public:
  /**
   * @brief Add an IMU measurement with explicit bias
   *
   * @param meas tuple containing measurement with explicit bias
   */
  void add_meas(const MeasBiased<T> & meas)
  {
    if (lastMeas_.has_value()) {
      const T dt = std::chrono::duration_cast<std::chrono::duration<T>>(
        std::get<timeIdx>(meas) - std::get<timeIdx>(lastMeas_.value())
        ).count();
      state_ = state_ * State<T>::exp(imu_kinematics(state_, lastMeas_.value()) * dt);
    } else {
      startTime_ = std::get<timeIdx>(meas);
    }
    lastMeas_ = meas;
  }

  /**
   * @brief Add an IMU measurement without bias
   *
   * @param t Time of measurement
   * @param accel Accelerometer measurement
   * @param gyro Gyroscope measurement
   */
  void add_meas(const Time t, const Accel & accel, const Gyro & gyro)
  {
    add_meas(
      std::forward_as_tuple(t, accel, gyro, AccelBias<T>::Zero(), GyroBias<T>::Zero())
    );
  }

  /**
   * @brief Retrieve the start time, end time, and relative pose of the object
   * Result is nullopt if <=1 measurements added
   * @return std::optional<std::tuple<Time, Time, State<T>>> Result
   */
  std::optional<std::tuple<Time, Time, State<T>>> retrieve()
  {
    if (lastMeas_.has_value()) {
      return std::make_tuple(startTime_, std::get<timeIdx>(lastMeas_.value()), state_);
    } else {
      return std::nullopt;
    }
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  State<T> state_{};
  std::optional<MeasBiased<T>> lastMeas_{};
  Time startTime_{};
};

/* -------------------------------------------------------------------------- */
/*                        IMU Relative Pose Computation                       */
/* -------------------------------------------------------------------------- */
// This approach is an implementation of the method described in
// "On-Manifold Preintegration for Real-Time Visual--Inertial Odometry" by Forster et al.

template<typename T = double>
Rotation<T> step_rotation(
  const Rotation<T> & rot, const Gyro & gyro, const GyroBias<T> & bias, const double dt
)
{
  return rot * Rotation<T>::exp((gyro - bias) * dt);
}

template<typename T = double>
Container<Rotation<T>> compute_relative_rotation_series(
  const Container<MeasBiased<T>> & measContainer)
{
  Rotation<T> relativeRot{};
  Container<Rotation<T>> ret;
  ret.push_back(relativeRot);
  for (auto it = measContainer.begin(); (it + 1) != measContainer.end(); ++it) {
    const double dt = std::chrono::duration_cast<std::chrono::duration<double>>(
      std::get<timeIdx>(*(it + 1)) - std::get<timeIdx>(*it)
      ).count();
    relativeRot =
      step_rotation<T>(relativeRot, std::get<gyroIdx>(*it), std::get<gyroBiasIdx>(*it), dt);
    ret.push_back(relativeRot);
  }
  return ret;
}

template<typename T = double>
Container<Velocity<T>> compute_relative_velocity_series(
  const Container<MeasBiased<T>> & measContainer)
{
  Velocity<T> relativeVel{};
  Container<Velocity<T>> ret;
  ret.push_back(relativeVel);
  for (std::size_t i = 0; i < measContainer.size() - 1; ++i) {
    const double dt = std::chrono::duration_cast<std::chrono::duration<double>>(
      std::get<timeIdx>(measContainer[i + 1]) - std::get<timeIdx>(measContainer[i])
      ).count();
    relativeVel = Velocity<T>(
      relativeVel.translation() +
      (std::get<accelIdx>(measContainer[i]) - std::get<accelBiasIdx>(measContainer[i])) * dt);
    ret.push_back(relativeVel);
  }
  return ret;
}

template<typename T = double>
Container<Position<T>> compute_relative_position(
  const Container<MeasBiased<T>> & measContainer, const Container<Rotation<T>> & rotContainer,
  const Container<Velocity<T>> & velContainer)
{
  Position<T> relativePos = Position<T>::Zero();
  Container<Position<T>> ret;
  ret.push_back(relativePos);
  for (std::size_t i = 0; i < measContainer.size() - 1; ++i) {
    const double dt = std::chrono::duration_cast<std::chrono::duration<double>>(
      std::get<timeIdx>(measContainer[i + 1]) - std::get<timeIdx>(measContainer[i])
      ).count();
    relativePos +=
      rotContainer[i] *
      (velContainer[i].translation() * dt +
      (std::get<accelIdx>(measContainer[i]) - std::get<accelBiasIdx>(measContainer[i])) * dt * dt /
      2);
    ret.push_back(relativePos);
  }
  return ret;
}

template<typename T = double>
State<T> compute_relative_state(const Container<MeasBiased<T>> & measContainer)
{
  Container<Rotation<T>> rotationSeries = compute_relative_rotation_series(measContainer);
  Container<Velocity<T>> velocitySeries = compute_relative_velocity_series(measContainer);
  Container<Position<T>> positionSeries =
    compute_relative_position(measContainer, rotationSeries, velocitySeries);

  return State<T>(Pose<T>(rotationSeries.back(), positionSeries.back()), velocitySeries.back());
}
}  // namespace bps::preint

#endif  // CBR_MATH__LIE__IMU_PREINTEGRATION_HPP_
