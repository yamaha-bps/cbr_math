// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_MATH__SOPHUS_HPP_
#define CBR_MATH__SOPHUS_HPP_

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "math.hpp"

namespace cbr
{

/***************************************************************************
 * \brief Convert SE2 pose to SE3 pose with undefined coordinates at 0
 ***************************************************************************/
template<typename T>
auto SE2toSE3(const Sophus::SE2<T> & se2) noexcept
{
  T yaw = se2.so2().log();
  return Sophus::SE3<T>(
    Eigen::Quaternion<T>(cos(yaw * .5), 0., 0., sin(yaw * .5)),
    {se2.translation()[0], se2.translation()[1], 0.});
}

/***************************************************************************
 * \brief Convert SE3 pose to SE2 pose
 ***************************************************************************/
template<typename T>
auto SE3toSE2(const Sophus::SE3<T> & se3) noexcept
{
  const auto & q = se3.unit_quaternion();

  return Sophus::SE2<T>(
    atan2(
      2. * (q.w() * q.z() + q.x() * q.y()),
      1. - 2. * (powFast<2>(q.y()) + powFast<2>(q.z()))),
    {se3.translation()[0], se3.translation()[1]});
}

}  // namespace cbr

#endif  // CBR_MATH__SOPHUS_HPP_
