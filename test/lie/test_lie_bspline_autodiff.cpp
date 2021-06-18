// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE


#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <autodiff/forward.hpp>
#include <autodiff/forward/eigen.hpp>

#include <sophus/se3.hpp>

#include <vector>
#include <limits>

#include "cbr_math/lie/bspline.hpp"
#include "cbr_math/lie/En.hpp"

using cbr::lie::bspline_detail::row_t;

constexpr double epsilon = 10 * std::numeric_limits<double>::epsilon();

TEST(BSpline, R1AutoDiff)
{
  using R1 = cbr::lie::En<double, 1, 0>;

  row_t breakPts = (row_t(2) << 1, 2).finished();
  std::vector<R1, Eigen::aligned_allocator<R1>> vals{R1{1}, R1{2}};
  cbr::lie::BSpline<R1, 1> spline(breakPts, vals);

  auto f = [&spline](const autodiff::dual & d) -> autodiff::dual {
      return spline.val<autodiff::forward::dual>(d).translation()(0);
    };

  autodiff::dual t = 1.;
  auto dfdt = autodiff::derivative(f, autodiff::wrt(t), autodiff::forward::at(t));
  ASSERT_NEAR(dfdt, 1, 1e-5);
  ASSERT_NEAR(dfdt, spline.der(1.)(0), 1e-5);

  t = 1.25;
  dfdt = autodiff::derivative(f, autodiff::wrt(t), autodiff::forward::at(t));
  ASSERT_NEAR(dfdt, 1, 1e-5);
  ASSERT_NEAR(dfdt, spline.der(1.25)(0), 1e-5);

  t = 1.75;
  dfdt = autodiff::derivative(f, autodiff::wrt(t), autodiff::forward::at(t));
  ASSERT_NEAR(dfdt, 1, 1e-5);
  ASSERT_NEAR(dfdt, spline.der(1.75)(0), 1e-5);

  t = 1.999;
  dfdt = autodiff::derivative(f, autodiff::wrt(t), autodiff::forward::at(t));
  ASSERT_NEAR(dfdt, 1, 1e-5);
  ASSERT_NEAR(dfdt, spline.der(1.999)(0), 1e-5);
}


TEST(BSpline, R1AutoDiffDeg3)
{
  using R1 = cbr::lie::En<double, 1, 0>;

  row_t breakPts = (row_t(4) << 1, 2, 2.5, 4).finished();
  std::vector<R1, Eigen::aligned_allocator<R1>> vals{R1{1}, R1{2}, R1{2}, R1{1}};
  cbr::lie::BSpline<R1, 3> spline(breakPts, vals);

  auto f = [&spline](const autodiff::dual & d) -> autodiff::dual {
      return spline.val<autodiff::forward::dual>(d).translation()(0);
    };

  autodiff::dual t = 1.;
  auto dfdt = autodiff::derivative(f, autodiff::wrt(t), autodiff::forward::at(t));
  ASSERT_NEAR(dfdt, spline.der(1.)(0), 1e-5);

  t = 1.25;
  dfdt = autodiff::derivative(f, autodiff::wrt(t), autodiff::forward::at(t));
  ASSERT_NEAR(dfdt, spline.der(1.25)(0), 1e-5);

  t = 1.75;
  dfdt = autodiff::derivative(f, autodiff::wrt(t), autodiff::forward::at(t));
  ASSERT_NEAR(dfdt, spline.der(1.75)(0), 1e-5);

  t = 1.999;
  dfdt = autodiff::derivative(f, autodiff::wrt(t), autodiff::forward::at(t));
  ASSERT_NEAR(dfdt, spline.der(1.999)(0), 1e-5);
}


TEST(BSpline, SO3AutoDiff)
{
  row_t breakPts = (row_t(2) << 1, 2).finished();
  std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>> vals{
    Sophus::SO3d::rotX(0.5), Sophus::SO3d::rotX(-0.5)
  };

  cbr::lie::BSpline spline(breakPts, vals, true);

  auto f = [&spline](const autodiff::VectorXdual & d) -> autodiff::Vector4dual {
      return spline.val<autodiff::forward::dual>(d(0)).unit_quaternion().coeffs().transpose();
    };

  autodiff::VectorXdual x{1};
  autodiff::VectorXdual F;

  x(0) = 1.1;  // beginning of spline, qx decreases and wx increases
  auto J = autodiff::forward::jacobian(f, autodiff::wrt(x), autodiff::forward::at(x), F);

  ASSERT_LE(J(0, 0), -0.05);
  ASSERT_NEAR(J(1, 0), 0, epsilon);
  ASSERT_NEAR(J(2, 0), 0, epsilon);
  ASSERT_GE(J(3, 0), 0.05);

  x(0) = 1.5;  // middle of spline, we are around zero rotation
  J = autodiff::forward::jacobian(f, autodiff::wrt(x), autodiff::forward::at(x), F);

  ASSERT_LE(J(0, 0), -0.05);
  ASSERT_NEAR(J(1, 0), 0, epsilon);
  ASSERT_NEAR(J(2, 0), 0, epsilon);
  ASSERT_NEAR(J(3, 0), 0, epsilon);
}


TEST(BSpline, SO3Dynamics)
{
  using Vector9d = Eigen::Matrix<autodiff::forward::dual, 9, 1>;

  std::default_random_engine rng(123);  // seed

  row_t breakPts = (row_t(6) << 1, 2, 3, 3.5, 4, 4.5).finished();
  std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>> vals{
    Sophus::SO3d::sampleUniform(rng), Sophus::SO3d::sampleUniform(rng),
    Sophus::SO3d::sampleUniform(rng), Sophus::SO3d::sampleUniform(rng),
    Sophus::SO3d::sampleUniform(rng), Sophus::SO3d::sampleUniform(rng)
  };

  cbr::lie::BSpline<Sophus::SO3d, 3> spline(breakPts, vals, false);

  double t = 1.5;

  auto Xmat = spline.val(t).matrix();

  autodiff::VectorXdual x{1};
  x(0) = t;
  Vector9d F_v;
  auto f = [&spline](const autodiff::VectorXdual & d) -> Vector9d {
      Vector9d ret;
      Eigen::Map<autodiff::Matrix3dual>(ret.data()) = spline.val<autodiff::dual>(d(0)).matrix();
      return ret;
    };
  auto dXmat_v = autodiff::forward::jacobian(f, autodiff::wrt(x), autodiff::forward::at(x), F_v);

  auto what = Sophus::SO3d::hat(spline.der(t));

  Eigen::Matrix3d F;
  Eigen::Map<Eigen::Matrix<double, 9, 1>>(F.data()) = F_v.cast<double>();
  Eigen::Matrix3d dXmat;
  Eigen::Map<Eigen::Matrix<double, 9, 1>>(dXmat.data()) = dXmat_v;

  ASSERT_LE((Xmat - F).norm(), 1e-8);
  ASSERT_LE((dXmat - Xmat * what).norm(), 1e-8);
}


TEST(BSpline, SE3Dynamics)
{
  using Vector16d = Eigen::Matrix<autodiff::forward::dual, 16, 1>;

  std::default_random_engine rng(123);  // seed

  row_t breakPts = (row_t(6) << 1, 2, 3, 3.5, 4, 4.5).finished();
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vals{
    Sophus::SE3d::sampleUniform(rng), Sophus::SE3d::sampleUniform(rng),
    Sophus::SE3d::sampleUniform(rng), Sophus::SE3d::sampleUniform(rng),
    Sophus::SE3d::sampleUniform(rng), Sophus::SE3d::sampleUniform(rng)
  };

  cbr::lie::BSpline<Sophus::SE3d, 3> spline(breakPts, vals, false);

  double t = 3.25;

  auto Xmat = spline.val(t).matrix();

  autodiff::VectorXdual x{1};
  x(0) = t;
  Vector16d F_v;
  auto f = [&spline](const autodiff::VectorXdual & d) -> Vector16d {
      Vector16d ret;
      Eigen::Map<autodiff::Matrix4dual>(ret.data()) = spline.val<autodiff::dual>(d(0)).matrix();
      return ret;
    };
  auto dXmat_v = autodiff::forward::jacobian(f, autodiff::wrt(x), autodiff::forward::at(x), F_v);

  auto what = Sophus::SE3d::hat(spline.der(t));

  Eigen::Matrix4d F;
  Eigen::Map<Eigen::Matrix<double, 16, 1>>(F.data()) = F_v.cast<double>();
  Eigen::Matrix4d dXmat;
  Eigen::Map<Eigen::Matrix<double, 16, 1>>(dXmat.data()) = dXmat_v;

  ASSERT_LE((Xmat - F).norm(), 1e-8);
  ASSERT_LE((dXmat - Xmat * what).norm(), 1e-8);
}

TEST(BSpline, CtrlPointDiff)
{
  using SO3T = Sophus::SO3<autodiff::dual>;
  using QuaternionT = Eigen::Quaternion<autodiff::dual>;

  row_t breakPts = (row_t(3) << 1, 2, 3).finished();

  auto f = [&breakPts](const autodiff::VectorXdual & x) {
      std::vector<SO3T, Eigen::aligned_allocator<SO3T>> vals{
        SO3T(Eigen::Map<const QuaternionT>(x.data())),
        SO3T(Eigen::Map<const QuaternionT>(x.data() + 4)),
        SO3T(Eigen::Map<const QuaternionT>(x.data() + 8))
      };

      cbr::lie::BSpline<Sophus::SO3<autodiff::dual>, 3> spline(breakPts, vals, false);

      return spline.val<autodiff::dual>(autodiff::dual(2.5)).params();
    };

  Eigen::Matrix<autodiff::dual, 12, 1> coeffs;
  coeffs.setZero();
  coeffs[3] = 1;
  coeffs[7] = 1;
  coeffs[11] = 1;

  auto jac = autodiff::forward::jacobian(
    f, autodiff::wrt(coeffs), autodiff::forward::at(coeffs)
  );
}
