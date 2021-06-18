// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE


#include <gtest/gtest.h>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <limits>
#include <vector>

#include "cbr_math/lie/En.hpp"
#include "cbr_math/lie/bspline.hpp"

using cbr::lie::bspline::coeffMatrix, cbr::lie::bspline::coeffMatrixCard;
using cbr::lie::bspline_detail::row_t;

constexpr double epsilon = 10 * std::numeric_limits<double>::epsilon();

TEST(BSpline, FindIndex1)
{
  row_t breakPts = Eigen::VectorXd::LinSpaced(11, 0, 10).transpose();
  ASSERT_EQ(cbr::lie::bspline::findIndex(0, breakPts), 0);
  ASSERT_EQ(cbr::lie::bspline::findIndex(0.99, breakPts), 0);
  ASSERT_EQ(cbr::lie::bspline::findIndex(1.99, breakPts), 1);
  ASSERT_EQ(cbr::lie::bspline::findIndex(5.99, breakPts), 5);
  ASSERT_EQ(cbr::lie::bspline::findIndex(8, breakPts), 8);
  ASSERT_EQ(cbr::lie::bspline::findIndex(8.99, breakPts), 8);
}

TEST(BSpline, FindIndex2)
{
  row_t breakPts = Eigen::VectorXd::LinSpaced(11, 0, 5).transpose();
  ASSERT_EQ(cbr::lie::bspline::findIndex(0, breakPts), 0);
  ASSERT_EQ(cbr::lie::bspline::findIndex(0.49, breakPts), 0);
  ASSERT_EQ(cbr::lie::bspline::findIndex(0.51, breakPts), 1);
  ASSERT_EQ(cbr::lie::bspline::findIndex(0.99, breakPts), 1);
  ASSERT_EQ(cbr::lie::bspline::findIndex(1.01, breakPts), 2);
}

TEST(BSpline, FindIndex3)
{
  row_t breakPts = Eigen::VectorXd(7);
  breakPts << 0, 0, 1, 1, 2, 3, 4;
  ASSERT_EQ(cbr::lie::bspline::findIndex(0, breakPts), 1);
  ASSERT_EQ(cbr::lie::bspline::findIndex(0.99, breakPts), 1);
  ASSERT_EQ(cbr::lie::bspline::findIndex(1.01, breakPts), 3);
  ASSERT_EQ(cbr::lie::bspline::findIndex(1.99, breakPts), 3);
}

TEST(BSpline, CoeffMatrix2)
{
  size_t n = 10, idx = 5;

  row_t breakPts = Eigen::VectorXd::LinSpaced(n + 1, 0, n).transpose();

  auto m = coeffMatrix<2>(idx, breakPts);
  ASSERT_NEAR(m(0, 0), 1. / 2, epsilon);
  ASSERT_NEAR(m(0, 1), 1. / 2, epsilon);
  ASSERT_NEAR(m(0, 2), 0. / 2, epsilon);
  ASSERT_NEAR(m(1, 0), -2. / 2, epsilon);
  ASSERT_NEAR(m(1, 1), 2. / 2, epsilon);
  ASSERT_NEAR(m(1, 2), 0. / 2, epsilon);
  ASSERT_NEAR(m(2, 0), 1. / 2, epsilon);
  ASSERT_NEAR(m(2, 1), -2. / 2, epsilon);
  ASSERT_NEAR(m(2, 2), 1. / 2, epsilon);

  m = coeffMatrixCard<2>();
  ASSERT_NEAR(m(0, 0), 1. / 2, epsilon);
  ASSERT_NEAR(m(0, 1), 1. / 2, epsilon);
  ASSERT_NEAR(m(0, 2), 0. / 2, epsilon);
  ASSERT_NEAR(m(1, 0), -2. / 2, epsilon);
  ASSERT_NEAR(m(1, 1), 2. / 2, epsilon);
  ASSERT_NEAR(m(1, 2), 0. / 2, epsilon);
  ASSERT_NEAR(m(2, 0), 1. / 2, epsilon);
  ASSERT_NEAR(m(2, 1), -2. / 2, epsilon);
  ASSERT_NEAR(m(2, 2), 1. / 2, epsilon);

  for (size_t i = 2; i != n - 1; ++i) {
    ASSERT_LE((coeffMatrix<2>(i, breakPts) - coeffMatrixCard<2>()).norm(), epsilon);
  }
}

TEST(BSpline, CoeffMatrix3)
{
  size_t n = 10, idx = 5;

  row_t breakPts = Eigen::VectorXd::LinSpaced(n + 1, 0, n).transpose();

  auto m = coeffMatrix<3>(idx, breakPts);
  ASSERT_NEAR(m(0, 0), 1. / 6, epsilon);
  ASSERT_NEAR(m(0, 1), 4. / 6, epsilon);
  ASSERT_NEAR(m(0, 2), 1. / 6, epsilon);
  ASSERT_NEAR(m(0, 3), 0. / 6, epsilon);
  ASSERT_NEAR(m(1, 0), -3. / 6, epsilon);
  ASSERT_NEAR(m(1, 1), 0. / 6, epsilon);
  ASSERT_NEAR(m(1, 2), 3. / 6, epsilon);
  ASSERT_NEAR(m(1, 3), 0. / 6, epsilon);
  ASSERT_NEAR(m(2, 0), 3. / 6, epsilon);
  ASSERT_NEAR(m(2, 1), -6. / 6, epsilon);
  ASSERT_NEAR(m(2, 2), 3. / 6, epsilon);
  ASSERT_NEAR(m(2, 3), 0. / 6, epsilon);
  ASSERT_NEAR(m(3, 0), -1. / 6, epsilon);
  ASSERT_NEAR(m(3, 1), 3. / 6, epsilon);
  ASSERT_NEAR(m(3, 2), -3. / 6, epsilon);
  ASSERT_NEAR(m(3, 3), 1. / 6, epsilon);

  m = coeffMatrixCard<3>();
  ASSERT_NEAR(m(0, 0), 1. / 6, epsilon);
  ASSERT_NEAR(m(0, 1), 4. / 6, epsilon);
  ASSERT_NEAR(m(0, 2), 1. / 6, epsilon);
  ASSERT_NEAR(m(0, 3), 0. / 6, epsilon);
  ASSERT_NEAR(m(1, 0), -3. / 6, epsilon);
  ASSERT_NEAR(m(1, 1), 0. / 6, epsilon);
  ASSERT_NEAR(m(1, 2), 3. / 6, epsilon);
  ASSERT_NEAR(m(1, 3), 0. / 6, epsilon);
  ASSERT_NEAR(m(2, 0), 3. / 6, epsilon);
  ASSERT_NEAR(m(2, 1), -6. / 6, epsilon);
  ASSERT_NEAR(m(2, 2), 3. / 6, epsilon);
  ASSERT_NEAR(m(2, 3), 0. / 6, epsilon);
  ASSERT_NEAR(m(3, 0), -1. / 6, epsilon);
  ASSERT_NEAR(m(3, 1), 3. / 6, epsilon);
  ASSERT_NEAR(m(3, 2), -3. / 6, epsilon);
  ASSERT_NEAR(m(3, 3), 1. / 6, epsilon);

  for (size_t i = 3; i != n - 2; ++i) {
    ASSERT_LE((coeffMatrix<3>(i, breakPts) - coeffMatrixCard<3>()).norm(), epsilon);
  }
}

TEST(BSpline, eval)
{
  using R1 = cbr::lie::En<double, 1, 0>;

  for (int i = 0; i != 2; ++i) {  // card and non-card
    bool card = static_cast<bool>(i);

    row_t breakPts = (row_t(6) << 1, 2, 3, 4, 5, 6).finished();
    std::vector<R1, Eigen::aligned_allocator<R1>> vals{R1{5}, R1{4}, R1{3.5},
      R1{3}, R1{2.5}, R1{2}};

    // order 1 spline: linear interpolation
    cbr::lie::BSpline<R1, 1> spline1(breakPts, vals, card);
    ASSERT_EQ(spline1.is_cardinal(), card);

    // test covers fcn
    ASSERT_TRUE(spline1.covers(1.));
    ASSERT_TRUE(spline1.covers(6. - epsilon));
    ASSERT_FALSE(spline1.covers(1. - epsilon));
    ASSERT_FALSE(spline1.covers(6.));

    ASSERT_NEAR(spline1.val(1.).translation()(0), 5, epsilon);
    ASSERT_NEAR(spline1.val(1.5).translation()(0), 4.5, epsilon);
    ASSERT_NEAR(spline1.val(2.).translation()(0), 4, epsilon);
    ASSERT_NEAR(spline1.val(3.).translation()(0), 3.5, epsilon);
    ASSERT_NEAR(spline1.val(3.25).translation()(0), 3.375, epsilon);
    ASSERT_NEAR(spline1.val(4.).translation()(0), 3, epsilon);
    ASSERT_NEAR(spline1.val(5.).translation()(0), 2.5, epsilon);
    ASSERT_NEAR(spline1.val(5.5).translation()(0), 2.25, epsilon);
    ASSERT_NEAR(spline1.val(6 - epsilon).translation()(0), 2, 10 * epsilon);

    // order 3 spline: cubic polynomials, will not be equal in general
    cbr::lie::BSpline<R1, 3> spline3(breakPts, vals, card);
    ASSERT_EQ(spline3.is_cardinal(), card);
    ASSERT_NEAR(spline3.val(1.).translation()(0), 5, epsilon);
    ASSERT_NEAR(spline3.val(2.).translation()(0), 4, 2e-1);
    ASSERT_NEAR(spline3.val(3.).translation()(0), 3.5, 1e-2);
    ASSERT_NEAR(spline3.val(4.).translation()(0), 3, 1e-2);
    ASSERT_NEAR(spline3.val(5.).translation()(0), 2.5, 1e-1);
    ASSERT_NEAR(spline3.val(6 - epsilon).translation()(0), 2, 10 * epsilon);
  }
}

TEST(BSpline, R2)
{
  using R2 = cbr::lie::En<double, 2, 0>;

  row_t breakPts = (row_t(4) << 1, 2, 4, 5).finished();
  std::vector<R2, Eigen::aligned_allocator<R2>> vals{R2{1, 3}, R2{2, 3}, R2{2, 4}, R2{3, 4}};
  cbr::lie::BSpline<R2, 3> spline(breakPts, vals, false);

  // non-uniform interval: expect non-cardinal
  ASSERT_FALSE(spline.is_cardinal());

  // expect it to be equal at end points
  ASSERT_LE((spline.val(1.).translation() - vals[0].translation()).norm(), epsilon);
  ASSERT_LE((spline.val(5 - epsilon).translation() - vals[3].translation()).norm(), 10 * epsilon);
}

TEST(BSpline, SO3)
{
  for (int i = 0; i != 2; ++i) {  // card and non-card
    bool card = static_cast<bool>(i);

    row_t breakPts = (row_t(2) << 1, 2).finished();
    std::vector<Sophus::SO3d, Eigen::aligned_allocator<Sophus::SO3d>> vals{
      Sophus::SO3d::rotZ(0.5), Sophus::SO3d::rotY(0.5)};

    cbr::lie::BSpline spline(breakPts, vals, card);
    ASSERT_EQ(spline.is_cardinal(), card);

    ASSERT_LE((vals[0].inverse() * spline.val(1.)).log().norm(), epsilon);
    ASSERT_LE((vals[1].inverse() * spline.val(2 - epsilon)).log().norm(), 10 * epsilon);

    vals[1] = Sophus::SO3d::rotZ(1.);
    spline = cbr::lie::BSpline(breakPts, vals, card);
    ASSERT_EQ(spline.is_cardinal(), card);

    ASSERT_LE((Sophus::SO3d::rotZ(0.5).inverse() * spline.val(1.)).log().norm(), epsilon);
    ASSERT_LE((Sophus::SO3d::rotZ(0.75).inverse() * spline.val(1.5)).log().norm(), epsilon);
    ASSERT_LE(
      (Sophus::SO3d::rotZ(1.).inverse() * spline.val(2 - epsilon)).log().norm(), 10 * epsilon);
  }
}

TEST(BSpline, SE3)
{
  for (int i = 0; i != 2; ++i) {  // card and non-card
    bool card = static_cast<bool>(i);

    row_t breakPts = (row_t(2) << 1, 2).finished();
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vals{
      Sophus::SE3d::trans(0, 0, 0), Sophus::SE3d::trans(1, 2, 3)};

    cbr::lie::BSpline spline(breakPts, vals, card);
    ASSERT_EQ(spline.is_cardinal(), card);
    ASSERT_LE((vals[0].inverse() * spline.val(1.)).log().norm(), epsilon);
    ASSERT_LE((Sophus::SE3d::trans(0.5, 1, 1.5).inverse() * spline.val(1.5)).log().norm(), epsilon);
    ASSERT_LE((vals[1].inverse() * spline.val(2 - epsilon)).log().norm(), 10 * epsilon);
  }
}

TEST(BSpline, SE3derivative)
{
  for (int i = 0; i != 2; ++i) {  // card and non-card
    bool card = static_cast<bool>(i);

    row_t breakPts = (row_t(3) << 1, 2, 3).finished();
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vals{
      Sophus::SE3d::trans(0, 0, 0), Sophus::SE3d::trans(1, 2, 3), Sophus::SE3d::trans(2, 4, 6)};

    cbr::lie::BSpline<Sophus::SE3d, 1> spline(breakPts, vals, card);
    ASSERT_EQ(spline.is_cardinal(), card);
    auto dxdt = spline.der<double>(1.5);
    ASSERT_LE((dxdt - (vals[0].inverse() * vals[1]).log()).norm(), 1e-5);
  }
}

TEST(BSpline, support)
{
  for (int i = 0; i != 2; ++i) {  // card and non-card
    bool card = static_cast<bool>(i);

    row_t breakPts = (row_t(2) << 1, 2).finished();
    std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vals{
      Sophus::SE3d::trans(0, 0, 0), Sophus::SE3d::trans(1, 2, 3)};

    cbr::lie::BSpline spline(breakPts, vals, card);
    ASSERT_DOUBLE_EQ(spline.support(1.5), 1);
  }
}

TEST(BSpline, D2SE2)
{
  std::default_random_engine rng(123);  // seed

  row_t breakPts = (row_t(6) << 1, 2, 3, 3.5, 4, 4.5).finished();
  std::vector<Sophus::SE2d, Eigen::aligned_allocator<Sophus::SE2d>> vals{
    Sophus::SE2d::sampleUniform(rng), Sophus::SE2d::sampleUniform(rng),
    Sophus::SE2d::sampleUniform(rng), Sophus::SE2d::sampleUniform(rng),
    Sophus::SE2d::sampleUniform(rng), Sophus::SE2d::sampleUniform(rng)};

  cbr::lie::BSpline<Sophus::SE2d, 5> spline(breakPts, vals, false);

  double h = 1e-5;

  for (double t = 1.25; t < 4.5; t += 0.25) {
    Eigen::Matrix3d X = spline.val(t).matrix();
    // numerically approximate derivatives since 2nd order ones seem tedious with autodiff
    Eigen::Matrix3d dX = (spline.val(t + h).matrix() - spline.val(t).matrix()) / h;
    Eigen::Matrix3d d2X =
      (spline.val(t - h).matrix() + spline.val(t + h).matrix() - 2 * spline.val(t).matrix()) /
      (h * h);

    Eigen::Matrix3d what = Sophus::SE2d::hat(spline.der(t));
    Eigen::Matrix3d dwhat = Sophus::SE2d::hat(spline.der2(t));

    ASSERT_LE((dX - X * what).norm(), 1e-3);
    ASSERT_LE((d2X - X * what * what - X * dwhat).norm(), 1e-3);
  }
}

TEST(BSpline, D2SE3)
{
  std::default_random_engine rng(123);  // seed

  row_t breakPts = (row_t(6) << 1, 2, 3, 3.5, 4, 4.5).finished();
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> vals{
    Sophus::SE3d::sampleUniform(rng), Sophus::SE3d::sampleUniform(rng),
    Sophus::SE3d::sampleUniform(rng), Sophus::SE3d::sampleUniform(rng),
    Sophus::SE3d::sampleUniform(rng), Sophus::SE3d::sampleUniform(rng)};

  cbr::lie::BSpline<Sophus::SE3d, 3> spline(breakPts, vals, false);

  double h = 1e-5;

  for (double t = 1.25; t < 4.5; t += 0.25) {
    auto X = spline.val(t).matrix();
    // numerically approximate derivatives since 2nd order ones seem tedious with autodiff
    Eigen::Matrix<double, 4, 4> dX = (spline.val(t + h).matrix() - spline.val(t).matrix()) / h;
    Eigen::Matrix<double, 4, 4> d2X =
      (spline.val(t - h).matrix() + spline.val(t + h).matrix() - 2 * spline.val(t).matrix()) /
      (h * h);

    auto what = Sophus::SE3d::hat(spline.der(t));
    auto dwhat = Sophus::SE3d::hat(spline.der2(t));

    ASSERT_LE((dX - X * what).norm(), 1e-3);
    ASSERT_LE((d2X - X * what * what - X * dwhat).norm(), 1e-3);
  }
}
