// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE


#include <gtest/gtest.h>

#include <sophus/se2.hpp>

#include <array>
#include <utility>
#include <vector>
#include <iomanip>

#include "cbr_math/interp.hpp"

constexpr double eps = 1e-9;

TEST(PiecewisePoly, Construct)
{
  using cbr::PiecewisePoly;

  PiecewisePoly::row_t bp(1, 1);
  PiecewisePoly::matrix_t coeffs(1, 1);
  ASSERT_ANY_THROW(PiecewisePoly pp(bp, coeffs));

  bp.resize(1, 2);
  coeffs.resize(1, 2);
  ASSERT_ANY_THROW(PiecewisePoly pp(bp, coeffs));

  bp.resize(1, 3);
  bp << 0., 1., 2.;
  coeffs.resize(1, 2);
  ASSERT_NO_THROW(PiecewisePoly pp(bp, coeffs));

  bp.resize(1, 4);
  bp << 0., 1., 2., 3.;
  coeffs.resize(1, 3);
  ASSERT_NO_THROW(PiecewisePoly pp(bp, coeffs));

  PiecewisePoly pp(bp, coeffs);
  auto lb = pp.lb();
  ASSERT_DOUBLE_EQ(lb, bp[0]);

  auto ub = pp.ub();
  ASSERT_DOUBLE_EQ(ub, bp[3]);

  const auto & [lbb, ubb] = pp.bounds();
  ASSERT_DOUBLE_EQ(lbb, lb);
  ASSERT_DOUBLE_EQ(ubb, ub);

  bp.resize(1, 4);
  coeffs.resize(1, 2);
  ASSERT_ANY_THROW(PiecewisePoly pp(bp, coeffs));

  bp.resize(1, 6);
  coeffs.resize(1, 5);
  bp << 0., 1., 2., 3., 4., 5.;
  ASSERT_NO_THROW(PiecewisePoly pp(bp, coeffs));

  bp << 0., 1., 1., 3., 4., 5.;
  ASSERT_ANY_THROW(PiecewisePoly pp(bp, coeffs));

  bp << 0., 2., 1., 3., 4., 5.;
  ASSERT_ANY_THROW(PiecewisePoly pp(bp, coeffs));

  coeffs.resize(0, 5);
  ASSERT_ANY_THROW(PiecewisePoly pp(bp, coeffs));
}

TEST(PiecewisePoly, Eval)
{
  using cbr::PiecewisePoly;
  PiecewisePoly::row_t bp(1, 7);
  bp << 0., 1., 3., 6., 10., 15., 21;

  PiecewisePoly::matrix_t coeffs(5, 6);
  coeffs <<
    0., 1., 2., 3., 4., 5.,
    5., 6., 3., 7., 2., 8.,
    8., 7., 6., 5., 4., 3.,
    1., 0., 2., 8., 3., 7.,
    6., 7., 4., 8., 9., 0.;


  std::array<PiecewisePoly::SEARCH, 4> ss;
  ss[0] = PiecewisePoly::SEARCH::LINEAR_FORWARD;
  ss[1] = PiecewisePoly::SEARCH::LINEAR_REVERSE;
  ss[2] = PiecewisePoly::SEARCH::LINEAR_HINT;
  ss[3] = PiecewisePoly::SEARCH::BINARY;

  for (const auto & s : ss) {
    bp << 0., 2., 4., 6., 8., 10., 12.;
    PiecewisePoly pp(bp, coeffs, PiecewisePoly::EXTRAP::CLAMP, s);

    ASSERT_DOUBLE_EQ(pp.val(1.), 20.);
    ASSERT_DOUBLE_EQ(pp.val(3.), 21.);
    ASSERT_DOUBLE_EQ(pp.val(5.), 17.);
    ASSERT_DOUBLE_EQ(pp.val(7.), 31.);
    ASSERT_DOUBLE_EQ(pp.val(9.), 22.);
    ASSERT_DOUBLE_EQ(pp.val(11.), 23.);

    ASSERT_DOUBLE_EQ(pp.val(1.), pp.der(1., 0));
    ASSERT_DOUBLE_EQ(pp.val(3.), pp.der(3., 0));
    ASSERT_DOUBLE_EQ(pp.val(5.), pp.der(5., 0));
    ASSERT_DOUBLE_EQ(pp.val(7.), pp.der(7., 0));
    ASSERT_DOUBLE_EQ(pp.val(9.), pp.der(9., 0));
    ASSERT_DOUBLE_EQ(pp.val(11.), pp.der(11., 0));

    ASSERT_DOUBLE_EQ(pp.der(1.), 32.);
    ASSERT_DOUBLE_EQ(pp.der(3.), 36.);
    ASSERT_DOUBLE_EQ(pp.der(5.), 31.);
    ASSERT_DOUBLE_EQ(pp.der(7.), 51.);
    ASSERT_DOUBLE_EQ(pp.der(9.), 33.);
    ASSERT_DOUBLE_EQ(pp.der(11.), 57.);

    ASSERT_DOUBLE_EQ(pp.der(1., 2), 46.);
    ASSERT_DOUBLE_EQ(pp.der(3., 2), 62.);
    ASSERT_DOUBLE_EQ(pp.der(5., 2), 54.);
    ASSERT_DOUBLE_EQ(pp.der(7., 2), 88.);
    ASSERT_DOUBLE_EQ(pp.der(9., 2), 68.);
    ASSERT_DOUBLE_EQ(pp.der(11., 2), 114.);

    ASSERT_DOUBLE_EQ(pp.der(1., 3), 30.);
    ASSERT_DOUBLE_EQ(pp.der(3., 3), 60.);
    ASSERT_DOUBLE_EQ(pp.der(5., 3), 66.);
    ASSERT_DOUBLE_EQ(pp.der(7., 3), 114.);
    ASSERT_DOUBLE_EQ(pp.der(9., 3), 108.);
    ASSERT_DOUBLE_EQ(pp.der(11., 3), 168.);

    bp << 0., 2., 5., 9., 14., 20., 27.;
    pp = PiecewisePoly(bp, coeffs, PiecewisePoly::EXTRAP::CLAMP, s);
    ASSERT_DOUBLE_EQ(pp.val(1.), 20.);
    ASSERT_DOUBLE_EQ(pp.val(4.), 99.);
    ASSERT_DOUBLE_EQ(pp.val(8.), 307.);
    ASSERT_DOUBLE_EQ(pp.val(13.), 1336.);
    ASSERT_DOUBLE_EQ(pp.val(19.), 2874.);
    ASSERT_DOUBLE_EQ(pp.val(26.), 8358.);

    ASSERT_DOUBLE_EQ(pp.val(1.), pp.der(1., 0));
    ASSERT_DOUBLE_EQ(pp.val(4.), pp.der(4., 0));
    ASSERT_DOUBLE_EQ(pp.val(8.), pp.der(8., 0));
    ASSERT_DOUBLE_EQ(pp.val(13.), pp.der(13., 0));
    ASSERT_DOUBLE_EQ(pp.val(19.), pp.der(19., 0));
    ASSERT_DOUBLE_EQ(pp.val(26.), pp.der(26., 0));

    ASSERT_DOUBLE_EQ(pp.der(1.), 32.);
    ASSERT_DOUBLE_EQ(pp.der(4.), 132.);
    ASSERT_DOUBLE_EQ(pp.der(8.), 335.);
    ASSERT_DOUBLE_EQ(pp.der(13.), 1152.);
    ASSERT_DOUBLE_EQ(pp.der(19.), 2193.);
    ASSERT_DOUBLE_EQ(pp.der(26.), 5227.);

    ASSERT_DOUBLE_EQ(pp.der(1., 2), 46.);
    ASSERT_DOUBLE_EQ(pp.der(4., 2), 134.);
    ASSERT_DOUBLE_EQ(pp.der(8., 2), 282.);
    ASSERT_DOUBLE_EQ(pp.der(13., 2), 754.);
    ASSERT_DOUBLE_EQ(pp.der(19., 2), 1268.);
    ASSERT_DOUBLE_EQ(pp.der(26., 2), 2454.);

    ASSERT_DOUBLE_EQ(pp.der(1., 3), 30.);
    ASSERT_DOUBLE_EQ(pp.der(4., 3), 84.);
    ASSERT_DOUBLE_EQ(pp.der(8., 3), 162.);
    ASSERT_DOUBLE_EQ(pp.der(13., 3), 330.);
    ASSERT_DOUBLE_EQ(pp.der(19., 3), 492.);
    ASSERT_DOUBLE_EQ(pp.der(26., 3), 768.);
  }

  bp << 0., 2., 4., 6., 8., 10., 12.;
  PiecewisePoly pp(bp, coeffs);

  Eigen::Matrix<double, 1, Eigen::Dynamic> r(1, 6);
  r << 1., 3., 5., 7., 9., 11.;
  auto rr = pp.vals(r);
  ASSERT_DOUBLE_EQ(pp.val(r[0]), rr[0]);
  ASSERT_DOUBLE_EQ(pp.val(r[1]), rr[1]);
  ASSERT_DOUBLE_EQ(pp.val(r[2]), rr[2]);
  ASSERT_DOUBLE_EQ(pp.val(r[3]), rr[3]);
  ASSERT_DOUBLE_EQ(pp.val(r[4]), rr[4]);
  ASSERT_DOUBLE_EQ(pp.val(r[5]), rr[5]);

  Eigen::Matrix<double, Eigen::Dynamic, 1> v(6, 1);
  v << 1., 3., 5., 7., 9., 11.;
  auto vv = pp.vals(v);
  ASSERT_DOUBLE_EQ(pp.val(v[0]), vv[0]);
  ASSERT_DOUBLE_EQ(pp.val(v[1]), vv[1]);
  ASSERT_DOUBLE_EQ(pp.val(v[2]), vv[2]);
  ASSERT_DOUBLE_EQ(pp.val(v[3]), vv[3]);
  ASSERT_DOUBLE_EQ(pp.val(v[4]), vv[4]);
  ASSERT_DOUBLE_EQ(pp.val(v[5]), vv[5]);

  Eigen::Matrix<double, 6, 1> v6;
  v6 << 1., 3., 5., 7., 9., 11.;
  auto vv6 = pp.vals(v6);
  ASSERT_DOUBLE_EQ(pp.val(v6[0]), vv6[0]);
  ASSERT_DOUBLE_EQ(pp.val(v6[1]), vv6[1]);
  ASSERT_DOUBLE_EQ(pp.val(v6[2]), vv6[2]);
  ASSERT_DOUBLE_EQ(pp.val(v6[3]), vv6[3]);
  ASSERT_DOUBLE_EQ(pp.val(v6[4]), vv6[4]);
  ASSERT_DOUBLE_EQ(pp.val(v6[5]), vv6[5]);

  Eigen::Matrix<double, 2, 6> v62;
  v62 << 1., 3., 5., 7., 9., 11.,
    1., 3., 5., 7., 9., 11.;
  auto vv62 = pp.vals(v62);
  ASSERT_DOUBLE_EQ(pp.val(v62(0, 0)), vv62(0, 0));
  ASSERT_DOUBLE_EQ(pp.val(v62(0, 1)), vv62(0, 1));
  ASSERT_DOUBLE_EQ(pp.val(v62(0, 2)), vv62(0, 2));
  ASSERT_DOUBLE_EQ(pp.val(v62(0, 3)), vv62(0, 3));
  ASSERT_DOUBLE_EQ(pp.val(v62(0, 4)), vv62(0, 4));
  ASSERT_DOUBLE_EQ(pp.val(v62(0, 5)), vv62(0, 5));
  ASSERT_DOUBLE_EQ(pp.val(v62(1, 0)), vv62(1, 0));
  ASSERT_DOUBLE_EQ(pp.val(v62(1, 1)), vv62(1, 1));
  ASSERT_DOUBLE_EQ(pp.val(v62(1, 2)), vv62(1, 2));
  ASSERT_DOUBLE_EQ(pp.val(v62(1, 3)), vv62(1, 3));
  ASSERT_DOUBLE_EQ(pp.val(v62(1, 4)), vv62(1, 4));
  ASSERT_DOUBLE_EQ(pp.val(v62(1, 5)), vv62(1, 5));

  Eigen::Map<Eigen::Matrix<double, 2, 6>> m62(v62.data());
  auto mm62 = pp.vals(m62);
  ASSERT_DOUBLE_EQ(pp.val(m62(0, 0)), mm62(0, 0));
  ASSERT_DOUBLE_EQ(pp.val(m62(0, 1)), mm62(0, 1));
  ASSERT_DOUBLE_EQ(pp.val(m62(0, 2)), mm62(0, 2));
  ASSERT_DOUBLE_EQ(pp.val(m62(0, 3)), mm62(0, 3));
  ASSERT_DOUBLE_EQ(pp.val(m62(0, 4)), mm62(0, 4));
  ASSERT_DOUBLE_EQ(pp.val(m62(0, 5)), mm62(0, 5));
  ASSERT_DOUBLE_EQ(pp.val(m62(1, 0)), mm62(1, 0));
  ASSERT_DOUBLE_EQ(pp.val(m62(1, 1)), mm62(1, 1));
  ASSERT_DOUBLE_EQ(pp.val(m62(1, 2)), mm62(1, 2));
  ASSERT_DOUBLE_EQ(pp.val(m62(1, 3)), mm62(1, 3));
  ASSERT_DOUBLE_EQ(pp.val(m62(1, 4)), mm62(1, 4));
  ASSERT_DOUBLE_EQ(pp.val(m62(1, 5)), mm62(1, 5));

  auto mm62d = pp.ders(m62);
  ASSERT_DOUBLE_EQ(pp.der(v62(0, 0)), mm62d(0, 0));
  ASSERT_DOUBLE_EQ(pp.der(v62(0, 1)), mm62d(0, 1));
  ASSERT_DOUBLE_EQ(pp.der(v62(0, 2)), mm62d(0, 2));
  ASSERT_DOUBLE_EQ(pp.der(v62(0, 3)), mm62d(0, 3));
  ASSERT_DOUBLE_EQ(pp.der(v62(0, 4)), mm62d(0, 4));
  ASSERT_DOUBLE_EQ(pp.der(v62(0, 5)), mm62d(0, 5));
  ASSERT_DOUBLE_EQ(pp.der(v62(1, 0)), mm62d(1, 0));
  ASSERT_DOUBLE_EQ(pp.der(v62(1, 1)), mm62d(1, 1));
  ASSERT_DOUBLE_EQ(pp.der(v62(1, 2)), mm62d(1, 2));
  ASSERT_DOUBLE_EQ(pp.der(v62(1, 3)), mm62d(1, 3));
  ASSERT_DOUBLE_EQ(pp.der(v62(1, 4)), mm62d(1, 4));
  ASSERT_DOUBLE_EQ(pp.der(v62(1, 5)), mm62d(1, 5));

  std::array<double, 6> xs{1., 3., 5., 7., 9., 11.};
  auto vals = pp.vals(xs);
  auto ders = pp.ders(xs);
  for (std::size_t i = 0; i < xs.size(); i++) {
    ASSERT_DOUBLE_EQ(pp.val(xs[i]), vals[i]);
    ASSERT_DOUBLE_EQ(pp.der(xs[i]), ders[i]);
  }

  std::array<float, 6> xsf{1., 3., 5., 7., 9., 11.};
  auto valsf = pp.vals(xsf);
  auto dersf = pp.ders(xsf);
  for (std::size_t i = 0; i < xsf.size(); i++) {
    ASSERT_FLOAT_EQ(pp.val(xsf[i]), valsf[i]);
    ASSERT_FLOAT_EQ(pp.der(xsf[i]), dersf[i]);
  }
}

TEST(PiecewisePoly, Extrap)
{
  using cbr::PiecewisePoly;
  PiecewisePoly::row_t bp(1, 3);
  bp << 0., 1., 3.;

  PiecewisePoly::matrix_t coeffs(3, 2);
  coeffs <<
    1., 0.,
    0., 1.,
    0., 1.;

  PiecewisePoly pp(bp, coeffs);

  pp.set_extrap(PiecewisePoly::EXTRAP::CLAMP);
  ASSERT_DOUBLE_EQ(pp.val(0.), 0.);
  ASSERT_DOUBLE_EQ(pp.val(-1.), 0.);
  ASSERT_DOUBLE_EQ(pp.val(-2.), 0.);
  ASSERT_DOUBLE_EQ(pp.val(3.), 3.);
  ASSERT_DOUBLE_EQ(pp.val(4.), 3.);
  ASSERT_DOUBLE_EQ(pp.val(5.), 3.);

  ASSERT_DOUBLE_EQ(pp.der(0.), 0.);
  ASSERT_DOUBLE_EQ(pp.der(-1.), 0.);
  ASSERT_DOUBLE_EQ(pp.der(-2.), 0.);
  ASSERT_DOUBLE_EQ(pp.der(3.), 0.);
  ASSERT_DOUBLE_EQ(pp.der(4.), 0.);
  ASSERT_DOUBLE_EQ(pp.der(5.), 0.);

  ASSERT_DOUBLE_EQ(pp.der(0., 2), 2.);
  ASSERT_DOUBLE_EQ(pp.der(-1., 2), 0.);
  ASSERT_DOUBLE_EQ(pp.der(-2., 2), 0.);
  ASSERT_DOUBLE_EQ(pp.der(3., 2), 0.);
  ASSERT_DOUBLE_EQ(pp.der(4., 2), 0.);
  ASSERT_DOUBLE_EQ(pp.der(5., 2), 0.);

  pp.set_extrap(PiecewisePoly::EXTRAP::LINEAR);
  ASSERT_DOUBLE_EQ(pp.val(0.), 0.);
  ASSERT_DOUBLE_EQ(pp.val(-1.), 0.);
  ASSERT_DOUBLE_EQ(pp.val(-2.), 0.);
  ASSERT_DOUBLE_EQ(pp.val(3.), 3.);
  ASSERT_DOUBLE_EQ(pp.val(4.), 4.);
  ASSERT_DOUBLE_EQ(pp.val(5.), 5.);

  ASSERT_DOUBLE_EQ(pp.der(0.), 0.);
  ASSERT_DOUBLE_EQ(pp.der(-1.), 0.);
  ASSERT_DOUBLE_EQ(pp.der(-2.), 0.);
  ASSERT_DOUBLE_EQ(pp.der(3.), 1.);
  ASSERT_DOUBLE_EQ(pp.der(4.), 1.);
  ASSERT_DOUBLE_EQ(pp.der(5.), 1.);

  ASSERT_DOUBLE_EQ(pp.der(0., 2), 2.);
  ASSERT_DOUBLE_EQ(pp.der(-1., 2), 0.);
  ASSERT_DOUBLE_EQ(pp.der(-2., 2), 0.);
  ASSERT_DOUBLE_EQ(pp.der(3., 2), 0.);
  ASSERT_DOUBLE_EQ(pp.der(4., 2), 0.);
  ASSERT_DOUBLE_EQ(pp.der(5., 2), 0.);

  pp.set_extrap(PiecewisePoly::EXTRAP::YOLO);
  ASSERT_DOUBLE_EQ(pp.val(0.), 0.);
  ASSERT_DOUBLE_EQ(pp.val(-1.), 1.);
  ASSERT_DOUBLE_EQ(pp.val(-2.), 4.);
  ASSERT_DOUBLE_EQ(pp.val(3.), 3.);
  ASSERT_DOUBLE_EQ(pp.val(4.), 4.);
  ASSERT_DOUBLE_EQ(pp.val(5.), 5.);

  ASSERT_DOUBLE_EQ(pp.der(0.), 0.);
  ASSERT_DOUBLE_EQ(pp.der(-1.), -2.);
  ASSERT_DOUBLE_EQ(pp.der(-2.), -4.);
  ASSERT_DOUBLE_EQ(pp.der(3.), 1.);
  ASSERT_DOUBLE_EQ(pp.der(4.), 1.);
  ASSERT_DOUBLE_EQ(pp.der(5.), 1.);

  ASSERT_DOUBLE_EQ(pp.der(0., 2), 2.);
  ASSERT_DOUBLE_EQ(pp.der(-1., 2), 2.);
  ASSERT_DOUBLE_EQ(pp.der(-2., 2), 2.);
  ASSERT_DOUBLE_EQ(pp.der(3., 2), 0.);
  ASSERT_DOUBLE_EQ(pp.der(4., 2), 0.);
  ASSERT_DOUBLE_EQ(pp.der(5., 2), 0.);
}

TEST(PiecewisePoly, PiecewisePolyND)
{
  using cbr::PiecewisePolyND;

  PiecewisePolyND::row_t bp(1, 7);
  bp << 0., 1., 3., 6., 10., 15., 21;

  PiecewisePolyND::matrix_t coeffs(1, 6);
  coeffs <<
    0., 1., 2., 3., 4., 5.;

  std::vector<PiecewisePolyND::matrix_t> coeffsList;
  coeffsList.reserve(3);

  for (std::size_t i = 0; i < 3; i++) {
    coeffsList.push_back(coeffs *= 2.);
  }

  PiecewisePolyND poly(bp, coeffsList);

  const auto val = poly.val(1.5);
  const auto der = poly.der(0.);

  double expectedVal = 1.;
  double expectedDer = 0.;
  for (std::size_t i = 0; i < poly.dimension(); i++) {
    expectedVal *= 2.;
    EXPECT_NEAR(val[i], expectedVal, eps);
    EXPECT_NEAR(der[i], expectedDer, eps);
  }

  bp.resize(1, 7);
  bp << 0., 1., 3., 6., 10., 15., 21;

  coeffs.resize(5, 6);
  coeffs <<
    0., 1., 2., 3., 4., 5.,
    5., 6., 3., 7., 2., 8.,
    8., 7., 6., 5., 4., 3.,
    1., 0., 2., 8., 3., 7.,
    6., 7., 4., 8., 9., 0.;

  coeffsList.clear();
  coeffsList.reserve(3);

  for (std::size_t i = 0; i < 3; i++) {
    coeffsList.push_back(coeffs *= 2.);
  }

  poly = PiecewisePolyND(bp, coeffsList);

  std::array xs{0.5, 3.5, 10.5};
  std::vector<double> xsv(xs.begin(), xs.end());
  Eigen::Map<Eigen::Matrix<double, 1, 3>> xsm(xs.data());

  const auto vals0 = poly.vals(xs);
  const auto vals1 = poly.vals(xsv);
  const auto vals2 = poly.vals(xsm);

  const auto ders0 = poly.ders(xs);
  const auto ders1 = poly.ders(xsv);
  const auto ders2 = poly.ders(xsm);

  for (std::size_t i = 0; i < poly.dimension(); i++) {
    for (std::size_t j = 0; j < xs.size(); j++) {
      EXPECT_NEAR(vals0[j][i], vals1[j][i], eps);
      EXPECT_NEAR(vals0[j][i], vals2(i, j), eps);

      EXPECT_NEAR(ders0[j][i], ders1[j][i], eps);
      EXPECT_NEAR(ders0[j][i], ders2(i, j), eps);
    }
  }
}

TEST(PiecewisePoly, PiecewiseConstantFit)
{
  using cbr::PiecewisePoly;
  PiecewisePoly::row_t x(6);
  x << 0., 1., 3., 4., 6., 8.;

  PiecewisePoly::row_t y1(6);
  y1 << 0., 1., 1., 0., -3., -1.;

  const auto pp = cbr::PiecewiseConstant::fit(x, y1);

  const std::vector<double> xs{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5};

  const auto vals = pp.vals(xs);
  ASSERT_DOUBLE_EQ(vals[0], 0.);
  ASSERT_DOUBLE_EQ(vals[1], 0.);
  ASSERT_DOUBLE_EQ(vals[2], 1.);
  ASSERT_DOUBLE_EQ(vals[3], 1.);
  ASSERT_DOUBLE_EQ(vals[4], 1.);
  ASSERT_DOUBLE_EQ(vals[5], 1.);
  ASSERT_DOUBLE_EQ(vals[6], 1.);
  ASSERT_DOUBLE_EQ(vals[7], 1.);
  ASSERT_DOUBLE_EQ(vals[8], 0.);
  ASSERT_DOUBLE_EQ(vals[9], 0.);
  ASSERT_DOUBLE_EQ(vals[10], 0.);
  ASSERT_DOUBLE_EQ(vals[11], 0.);
  ASSERT_DOUBLE_EQ(vals[12], -3.);
  ASSERT_DOUBLE_EQ(vals[13], -3.);
  ASSERT_DOUBLE_EQ(vals[14], -3.);
  ASSERT_DOUBLE_EQ(vals[15], -3.);
  ASSERT_DOUBLE_EQ(vals[16], -1.);
  ASSERT_DOUBLE_EQ(vals[17], -1.);

  const auto ders = pp.ders(xs);
  ASSERT_DOUBLE_EQ(ders[0], 0.);
  ASSERT_DOUBLE_EQ(ders[1], 0.);
  ASSERT_DOUBLE_EQ(ders[2], 0.);
  ASSERT_DOUBLE_EQ(ders[3], 0.);
  ASSERT_DOUBLE_EQ(ders[4], 0.);
  ASSERT_DOUBLE_EQ(ders[5], 0.);
  ASSERT_DOUBLE_EQ(ders[6], 0.);
  ASSERT_DOUBLE_EQ(ders[7], 0.);
  ASSERT_DOUBLE_EQ(ders[8], 0.);
  ASSERT_DOUBLE_EQ(ders[9], 0.);
  ASSERT_DOUBLE_EQ(ders[10], 0.);
  ASSERT_DOUBLE_EQ(ders[11], 0.);
  ASSERT_DOUBLE_EQ(ders[12], 0.);
  ASSERT_DOUBLE_EQ(ders[13], 0.);
  ASSERT_DOUBLE_EQ(ders[14], 0.);
  ASSERT_DOUBLE_EQ(ders[15], 0.);
  ASSERT_DOUBLE_EQ(ders[16], 0.);
  ASSERT_DOUBLE_EQ(ders[17], 0.);

  const auto ders2 = pp.ders(xs, 2);
  for (const auto & der : ders2) {
    ASSERT_DOUBLE_EQ(der, 0.);
  }

  // Multidimensional
  const PiecewisePoly::row_t y2 = y1 + PiecewisePoly::row_t::Constant(1, 6, 1.);
  const PiecewisePoly::row_t y3 = y1 * 2.;

  PiecewisePoly::matrix_t ym(3, 6);
  ym.row(0) = y1;
  ym.row(1) = y2;
  ym.row(2) = y3;

  std::vector yv{y1, y2, y3};
  std::array ya{y1, y2, y3};

  const auto ppND = cbr::PiecewiseConstant::fitND(x, ym);
  const auto ppNDv = cbr::PiecewiseConstant::fitND(x, yv);
  const auto ppNDa = cbr::PiecewiseConstant::fitND(x, ya);

  const Eigen::Map<const PiecewisePoly::row_t> xsm(xs.data(), 1, xs.size());
  const auto valss = ppND.vals(xsm);
  const auto valssv = ppNDv.vals(xsm);
  const auto valssa = ppNDa.vals(xsm);

  for (std::size_t j = 0; j < xs.size(); j++) {
    for (std::size_t i = 0; i < ppND.dimension(); i++) {
      ASSERT_DOUBLE_EQ(valss(i, j), valssv(i, j));
      ASSERT_DOUBLE_EQ(valss(i, j), valssa(i, j));
    }

    ASSERT_DOUBLE_EQ(valss(1, j), valss(0, j) + 1.);
    ASSERT_DOUBLE_EQ(valss(2, j), valss(0, j) * 2.);
  }

  // Single value scalar
  PiecewisePoly::row_t x_SS(1);
  PiecewisePoly::row_t y_SS(1);

  x_SS(0) = 0.5;
  y_SS(0) = 3;

  const auto ppSS = cbr::PiecewiseConstant::fit(x_SS, y_SS);

  for (double d = 0.51; d < 6; ++d) {
    ASSERT_DOUBLE_EQ(ppSS.val(d), 3);
  }

  // Single value vector
  PiecewisePoly::row_t x_SV(1);
  PiecewisePoly::matrix_t y_SV(2, 1);

  x_SV(0) = 0.5;
  y_SV(0, 0) = 3;
  y_SV(1, 0) = -3;

  const auto ppSV = cbr::PiecewiseConstant::fitND(x_SV, y_SV);

  for (double d = 0.51; d < 6; ++d) {
    ASSERT_DOUBLE_EQ(ppSV.val(d)(0), 3);
    ASSERT_DOUBLE_EQ(ppSV.val(d)(1), -3);
  }
}

TEST(PiecewisePoly, PiecewiseLinearFit)
{
  using cbr::PiecewisePoly;
  PiecewisePoly::row_t x(6);
  x << 0., 1., 3., 4., 6., 8.;

  PiecewisePoly::row_t y1(6);
  y1 << 0., 1., 1., 0., -3., -1.;

  const auto pp = cbr::PiecewiseLinear::fit(x, y1);


  const std::vector<double> xs{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0, 7.5, 8.0};

  const auto vals = pp.vals(xs);
  ASSERT_DOUBLE_EQ(vals[0], 0.);
  ASSERT_DOUBLE_EQ(vals[1], 0.5);
  ASSERT_DOUBLE_EQ(vals[2], 1.);
  ASSERT_DOUBLE_EQ(vals[3], 1.);
  ASSERT_DOUBLE_EQ(vals[4], 1.);
  ASSERT_DOUBLE_EQ(vals[5], 1.);
  ASSERT_DOUBLE_EQ(vals[6], 1.);
  ASSERT_DOUBLE_EQ(vals[7], 0.5);
  ASSERT_DOUBLE_EQ(vals[8], 0.);
  ASSERT_DOUBLE_EQ(vals[9], -0.75);
  ASSERT_DOUBLE_EQ(vals[10], -1.5);
  ASSERT_DOUBLE_EQ(vals[11], -2.25);
  ASSERT_DOUBLE_EQ(vals[12], -3.);
  ASSERT_DOUBLE_EQ(vals[13], -2.5);
  ASSERT_DOUBLE_EQ(vals[14], -2.);
  ASSERT_DOUBLE_EQ(vals[15], -1.5);
  ASSERT_DOUBLE_EQ(vals[16], -1.);

  const auto ders = pp.ders(xs);
  ASSERT_DOUBLE_EQ(ders[0], 1.);
  ASSERT_DOUBLE_EQ(ders[1], 1.);
  ASSERT_DOUBLE_EQ(ders[2], 0.);
  ASSERT_DOUBLE_EQ(ders[3], 0.);
  ASSERT_DOUBLE_EQ(ders[4], 0.);
  ASSERT_DOUBLE_EQ(ders[5], 0.);
  ASSERT_DOUBLE_EQ(ders[6], -1.);
  ASSERT_DOUBLE_EQ(ders[7], -1);
  ASSERT_DOUBLE_EQ(ders[8], -1.5);
  ASSERT_DOUBLE_EQ(ders[9], -1.5);
  ASSERT_DOUBLE_EQ(ders[10], -1.5);
  ASSERT_DOUBLE_EQ(ders[11], -1.5);
  ASSERT_DOUBLE_EQ(ders[12], 1.);
  ASSERT_DOUBLE_EQ(ders[13], 1.);
  ASSERT_DOUBLE_EQ(ders[14], 1.);
  ASSERT_DOUBLE_EQ(ders[15], 1.);
  ASSERT_DOUBLE_EQ(ders[16], 0.);

  const auto ders2 = pp.ders(xs, 2);
  for (const auto & der : ders2) {
    ASSERT_DOUBLE_EQ(der, 0.);
  }

  // Multidimensional
  const PiecewisePoly::row_t y2 = y1 + PiecewisePoly::row_t::Constant(1, 6, 1.);
  const PiecewisePoly::row_t y3 = y1 * 2.;

  PiecewisePoly::matrix_t ym(3, 6);
  ym.row(0) = y1;
  ym.row(1) = y2;
  ym.row(2) = y3;

  std::vector yv{y1, y2, y3};
  std::array ya{y1, y2, y3};

  const auto ppND = cbr::PiecewiseLinear::fitND(x, ym);
  const auto ppNDv = cbr::PiecewiseLinear::fitND(x, yv);
  const auto ppNDa = cbr::PiecewiseLinear::fitND(x, ya);

  const Eigen::Map<const PiecewisePoly::row_t> xsm(xs.data(), 1, xs.size());
  const auto valss = ppND.vals(xsm);
  const auto valssv = ppNDv.vals(xsm);
  const auto valssa = ppNDa.vals(xsm);

  for (std::size_t j = 0; j < xs.size(); j++) {
    for (std::size_t i = 0; i < ppND.dimension(); i++) {
      ASSERT_DOUBLE_EQ(valss(i, j), valssv(i, j));
      ASSERT_DOUBLE_EQ(valss(i, j), valssa(i, j));
    }

    ASSERT_DOUBLE_EQ(valss(1, j), valss(0, j) + 1.);
    ASSERT_DOUBLE_EQ(valss(2, j), valss(0, j) * 2.);
  }
}

TEST(PiecewisePoly, SplineFit)
{
  using cbr::PiecewisePoly;
  PiecewisePoly::row_t x(1, 6);
  x << 0., 1., 3., 4., 6., 8.;
  PiecewisePoly::matrix_t y1(3, 6);
  y1 << 0., 1., 1., 0., -3., -1.,
    1., 2., 3., 4., 5., 6.,
    6., 5., 4., 3., 2., 1.;

  std::vector<double> xs{0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    5.5, 6.0, 6.5, 7.0, 7.5, 8.0 - 1e-3 * eps};

  auto pp = cbr::Spline::fit(x, y1.topRows(1));

  auto od = pp.order();
  auto le = pp.length();

  ASSERT_EQ(od, 4LU);
  ASSERT_EQ(le, 6LU);

  for (int64_t i = 0; i < x.cols(); i++) {
    EXPECT_NEAR(y1(0, i), pp.val(x[i]), eps);
  }

  auto vals = pp.vals(xs);
  EXPECT_NEAR(vals[0], 0.0000000000, eps);
  EXPECT_NEAR(vals[1], 0.5849309816, eps);
  EXPECT_NEAR(vals[2], 1.0000000000, eps);
  EXPECT_NEAR(vals[3], 1.2471242331, eps);
  EXPECT_NEAR(vals[4], 1.3282208589, eps);
  EXPECT_NEAR(vals[5], 1.2452070552, eps);
  EXPECT_NEAR(vals[6], 1.0000000000, eps);
  EXPECT_NEAR(vals[7], 0.5906825153, eps);
  EXPECT_NEAR(vals[8], 0.0000000000, eps);
  EXPECT_NEAR(vals[9], -0.7681652607, eps);
  EXPECT_NEAR(vals[10], -1.6100460123, eps);
  EXPECT_NEAR(vals[11], -2.3969037577, eps);
  EXPECT_NEAR(vals[12], -3.0000000000, eps);
  EXPECT_NEAR(vals[13], -3.2905962423, eps);
  EXPECT_NEAR(vals[14], -3.1399539877, eps);
  EXPECT_NEAR(vals[15], -2.4193347393, eps);
  EXPECT_NEAR(vals[16], -1.0000000000, eps);

  auto ders = pp.ders(xs);
  EXPECT_NEAR(ders[0], 1.3410020450, eps);
  EXPECT_NEAR(ders[1], 0.9993609407, eps);
  EXPECT_NEAR(ders[2], 0.6615541922, eps);
  EXPECT_NEAR(ders[3], 0.3275817996, eps);
  EXPECT_NEAR(ders[4], -0.0025562372, eps);
  EXPECT_NEAR(ders[5], -0.3288599182, eps);
  EXPECT_NEAR(ders[6], -0.6513292434, eps);
  EXPECT_NEAR(ders[7], -0.9929703476, eps);
  EXPECT_NEAR(ders[8], -1.3767893661, eps);
  EXPECT_NEAR(ders[9], -1.6529588446, eps);
  EXPECT_NEAR(ders[10], -1.6716513292, eps);
  EXPECT_NEAR(ders[11], -1.4328668200, eps);
  EXPECT_NEAR(ders[12], -0.9366053170, eps);
  EXPECT_NEAR(ders[13], -0.1828668200, eps);
  EXPECT_NEAR(ders[14], 0.8283486708, eps);
  EXPECT_NEAR(ders[15], 2.0970411554, eps);
  EXPECT_NEAR(ders[16], 3.6232106339, eps);

  auto ders2 = pp.ders(xs, 2);
  EXPECT_NEAR(ders2[0], -0.6871165644, eps);
  EXPECT_NEAR(ders2[1], -0.6794478528, eps);
  EXPECT_NEAR(ders2[2], -0.6717791411, eps);
  EXPECT_NEAR(ders2[3], -0.6641104294, eps);
  EXPECT_NEAR(ders2[4], -0.6564417178, eps);
  EXPECT_NEAR(ders2[5], -0.6487730061, eps);
  EXPECT_NEAR(ders2[6], -0.6411042945, eps);
  EXPECT_NEAR(ders2[7], -0.7254601227, eps);
  EXPECT_NEAR(ders2[8], -0.8098159509, eps);
  EXPECT_NEAR(ders2[9], -0.2948619632, eps);
  EXPECT_NEAR(ders2[10], 0.2200920245, eps);
  EXPECT_NEAR(ders2[11], 0.7350460123, eps);
  EXPECT_NEAR(ders2[12], 1.2500000000, eps);
  EXPECT_NEAR(ders2[13], 1.7649539877, eps);
  EXPECT_NEAR(ders2[14], 2.2799079755, eps);
  EXPECT_NEAR(ders2[15], 2.7948619632, eps);
  EXPECT_NEAR(ders2[16], 3.3098159509, eps);

  // With derivative
  pp = cbr::Spline::fit(x, y1.topRows(2));

  od = pp.order();
  le = pp.length();

  ASSERT_EQ(od, 4LU);
  ASSERT_EQ(le, 6LU);

  for (int64_t i = 0; i < x.cols() - 1; i++) {
    EXPECT_NEAR(y1(0, i), pp.val(x[i]), eps);
    EXPECT_NEAR(y1(1, i), pp.der(x[i], 1), eps);
  }

  vals = pp.vals(xs);
  EXPECT_NEAR(vals[0], 0.0000000000, eps);
  EXPECT_NEAR(vals[1], 0.3750000000, eps);
  EXPECT_NEAR(vals[2], 1.0000000000, eps);
  EXPECT_NEAR(vals[3], 1.2812500000, eps);
  EXPECT_NEAR(vals[4], 0.7500000000, eps);
  EXPECT_NEAR(vals[5], 0.3437500000, eps);
  EXPECT_NEAR(vals[6], 1.0000000000, eps);
  EXPECT_NEAR(vals[7], 0.3750000000, eps);
  EXPECT_NEAR(vals[8], 0.0000000000, eps);
  EXPECT_NEAR(vals[9], 0.1875000000, eps);
  EXPECT_NEAR(vals[10], -1.7500000000, eps);
  EXPECT_NEAR(vals[11], -3.5625000000, eps);
  EXPECT_NEAR(vals[12], -3.0000000000, eps);
  EXPECT_NEAR(vals[13], -1.8437500000, eps);
  EXPECT_NEAR(vals[14], -2.2500000000, eps);
  EXPECT_NEAR(vals[15], -2.5312500000, eps);
  EXPECT_NEAR(vals[16], -1.0000000000, eps);

  ders = pp.ders(xs);
  EXPECT_NEAR(ders[0], 1.0000000000, eps);
  EXPECT_NEAR(ders[1], 0.7500000000, eps);
  EXPECT_NEAR(ders[2], 2.0000000000, eps);
  EXPECT_NEAR(ders[3], -0.5625000000, eps);
  EXPECT_NEAR(ders[4], -1.2500000000, eps);
  EXPECT_NEAR(ders[5], -0.0625000000, eps);
  EXPECT_NEAR(ders[6], 3.0000000000, eps);
  EXPECT_NEAR(ders[7], -3.2500000000, eps);
  EXPECT_NEAR(ders[8], 4.0000000000, eps);
  EXPECT_NEAR(ders[9], -2.5000000000, eps);
  EXPECT_NEAR(ders[10], -4.5000000000, eps);
  EXPECT_NEAR(ders[11], -2.0000000000, eps);
  EXPECT_NEAR(ders[12], 5.0000000000, eps);
  EXPECT_NEAR(ders[13], 0.1875000000, eps);
  EXPECT_NEAR(ders[14], -1.2500000000, eps);
  EXPECT_NEAR(ders[15], 0.6875000000, eps);
  EXPECT_NEAR(ders[16], 6.0000000000, eps);

  ders2 = pp.ders(xs, 2);
  EXPECT_NEAR(ders2[0], -2.0000000000, eps);
  EXPECT_NEAR(ders2[1], 1.0000000000, eps);
  EXPECT_NEAR(ders2[2], -7.0000000000, eps);
  EXPECT_NEAR(ders2[3], -3.2500000000, eps);
  EXPECT_NEAR(ders2[4], 0.5000000000, eps);
  EXPECT_NEAR(ders2[5], 4.2500000000, eps);
  EXPECT_NEAR(ders2[6], -26.0000000000, eps);
  EXPECT_NEAR(ders2[7], 1.0000000000, eps);
  EXPECT_NEAR(ders2[8], -17.5000000000, eps);
  EXPECT_NEAR(ders2[9], -8.5000000000, eps);
  EXPECT_NEAR(ders2[10], 0.5000000000, eps);
  EXPECT_NEAR(ders2[11], 9.5000000000, eps);
  EXPECT_NEAR(ders2[12], -13.0000000000, eps);
  EXPECT_NEAR(ders2[13], -6.2500000000, eps);
  EXPECT_NEAR(ders2[14], 0.5000000000, eps);
  EXPECT_NEAR(ders2[15], 7.2500000000, eps);
  EXPECT_NEAR(ders2[16], 14.0000000000, eps);

  // With first and second derivatives
  pp = cbr::Spline::fit(x, y1);

  od = pp.order();
  le = pp.length();

  ASSERT_EQ(od, 6LU);
  ASSERT_EQ(le, 6LU);

  for (int64_t i = 0; i < x.cols() - 1; i++) {
    EXPECT_NEAR(y1(0, i), pp.val(x[i]), eps);
    EXPECT_NEAR(y1(1, i), pp.der(x[i], 1), eps);
    EXPECT_NEAR(y1(2, i), pp.der(x[i], 2), eps);
  }

  vals = pp.vals(xs);
  EXPECT_NEAR(vals[0], 0.0000000000, eps);
  EXPECT_NEAR(vals[1], 0.5156250000, eps);
  EXPECT_NEAR(vals[2], 1.0000000000, eps);
  EXPECT_NEAR(vals[3], 1.8437500000, eps);
  EXPECT_NEAR(vals[4], 1.2500000000, eps);
  EXPECT_NEAR(vals[5], 0.3437500000, eps);
  EXPECT_NEAR(vals[6], 1.0000000000, eps);
  EXPECT_NEAR(vals[7], 0.4531250000, eps);
  EXPECT_NEAR(vals[8], 0.0000000000, eps);
  EXPECT_NEAR(vals[9], 0.9785156250, eps);
  EXPECT_NEAR(vals[10], -1.5000000000, eps);
  EXPECT_NEAR(vals[11], -4.0722656250, eps);
  EXPECT_NEAR(vals[12], -3.0000000000, eps);
  EXPECT_NEAR(vals[13], -1.2812500000, eps);
  EXPECT_NEAR(vals[14], -2.1250000000, eps);
  EXPECT_NEAR(vals[15], -2.9531250000, eps);
  EXPECT_NEAR(vals[16], -1.0000000000, eps);

  ders = pp.ders(xs);
  EXPECT_NEAR(ders[0], 1.0000000000, eps);
  EXPECT_NEAR(ders[1], 0.5312500000, eps);
  EXPECT_NEAR(ders[2], 2.0000000000, eps);
  EXPECT_NEAR(ders[3], 0.3750000000, eps);
  EXPECT_NEAR(ders[4], -2.2500000000, eps);
  EXPECT_NEAR(ders[5], -0.6250000000, eps);
  EXPECT_NEAR(ders[6], 3.0000000000, eps);
  EXPECT_NEAR(ders[7], -4.9687500000, eps);
  EXPECT_NEAR(ders[8], 4.0000000000, eps);
  EXPECT_NEAR(ders[9], -1.6914062500, eps);
  EXPECT_NEAR(ders[10], -6.8125000000, eps);
  EXPECT_NEAR(ders[11], -1.9414062500, eps);
  EXPECT_NEAR(ders[12], 5.0000000000, eps);
  EXPECT_NEAR(ders[13], 0.7031250000, eps);
  EXPECT_NEAR(ders[14], -3.0000000000, eps);
  EXPECT_NEAR(ders[15], 0.8281250000, eps);
  EXPECT_NEAR(ders[16], 6.0000000000, eps);

  ders2 = pp.ders(xs, 2);
  EXPECT_NEAR(ders2[0], 6.0000000000, eps);
  EXPECT_NEAR(ders2[1], -1.2500000000, eps);
  EXPECT_NEAR(ders2[2], 5.0000000000, eps);
  EXPECT_NEAR(ders2[3], -7.2500000000, eps);
  EXPECT_NEAR(ders2[4], -1.5000000000, eps);
  EXPECT_NEAR(ders2[5], 7.2500000000, eps);
  EXPECT_NEAR(ders2[6], 4.0000000000, eps);
  EXPECT_NEAR(ders2[7], -0.2500000000, eps);
  EXPECT_NEAR(ders2[8], 3.0000000000, eps);
  EXPECT_NEAR(ders2[9], -16.8437500000, eps);
  EXPECT_NEAR(ders2[10], -0.5000000000, eps);
  EXPECT_NEAR(ders2[11], 17.3437500000, eps);
  EXPECT_NEAR(ders2[12], 2.0000000000, eps);
  EXPECT_NEAR(ders2[13], -12.5000000000, eps);
  EXPECT_NEAR(ders2[14], 0.0000000000, eps);
  EXPECT_NEAR(ders2[15], 13.2500000000, eps);
  EXPECT_NEAR(ders2[16], 1.0000000000, eps);

  x.resize(1, 3);
  x << 1., 2., 3.;
  y1.resize(1, 3);
  pp = cbr::Spline::fit(x, y1);
  od = pp.order();
  ASSERT_EQ(od, 3LU);

  x.resize(1, 2);
  x << 1., 2.;
  y1.resize(1, 2);
  pp = cbr::Spline::fit(x, y1);
  od = pp.order();
  ASSERT_EQ(od, 2LU);

  // Multidimensional
  x.resize(1, 6);
  x << 0., 1., 3., 4., 6., 8.;
  y1.resize(3, 6);
  y1 << 0., 1., 1., 0., -3., -1.,
    1., 2., 3., 4., 5., 6.,
    6., 5., 4., 3., 2., 1.;

  const PiecewisePoly::row_t y2 = y1.row(0) + PiecewisePoly::row_t::Constant(1, 6, 1.);
  const PiecewisePoly::row_t y3 = y1.row(0) * 2.;

  PiecewisePoly::matrix_t ym(3, 6);
  ym.row(0) = y1.row(0);
  ym.row(1) = y2;
  ym.row(2) = y3;

  std::vector<PiecewisePoly::row_t> yv{y1.row(0), y2, y3};
  std::array<PiecewisePoly::row_t, 3> ya{y1.row(0), y2, y3};

  PiecewisePoly::matrix_t oft = PiecewisePoly::matrix_t::Zero(y1.rows(), y1.cols());
  oft.row(0) = PiecewisePoly::row_t::Constant(1, y1.cols(), 1.);

  PiecewisePoly::matrix_t oft2 = PiecewisePoly::matrix_t::Ones(y1.rows(), y1.cols());
  oft2.row(0) *= 2.;

  std::vector<PiecewisePoly::matrix_t> yd{
    y1,
    y1 + oft,
    (y1.array() * oft2.array()).matrix()};

  const auto ppND = cbr::Spline::fitND(x, ym);
  const auto ppNDv = cbr::Spline::fitND(x, yv);
  const auto ppNDa = cbr::Spline::fitND(x, ya);
  const auto ppNDd = cbr::Spline::fitND(x, yd);

  const Eigen::Map<const PiecewisePoly::row_t> xsm(xs.data(), 1, xs.size());
  const auto valss = ppND.vals(xsm);
  const auto valssv = ppNDv.vals(xsm);
  const auto valssa = ppNDa.vals(xsm);

  for (std::size_t j = 0; j < xs.size(); j++) {
    for (std::size_t i = 0; i < ppND.dimension(); i++) {
      EXPECT_NEAR(valss(i, j), valssv(i, j), eps);
      EXPECT_NEAR(valss(i, j), valssa(i, j), eps);
    }

    EXPECT_NEAR(valss(1, j), valss(0, j) + 1., eps);
    EXPECT_NEAR(valss(2, j), valss(0, j) * 2., eps);
  }

  const auto valssd = ppND.vals(x);
  const auto valssdd = ppNDd.vals(x);

  for (Eigen::Index j = 0; j < x.size(); j++) {
    for (std::size_t i = 0; i < ppND.dimension(); i++) {
      EXPECT_NEAR(valssd(i, j), valssdd(i, j), eps);
    }
    EXPECT_NEAR(valssdd(1, j), valssdd(0, j) + 1., eps);
    EXPECT_NEAR(valssdd(2, j), valssdd(0, j) * 2., eps);
  }
}

template<typename group_t>
struct LieGroupTest
{
  using Tangent = typename group_t::Tangent;
  constexpr static int Dimension = group_t::DoF;

  void operator()()
  {
    using cbr::PiecewisePolyLie;
    using cbr::PiecewisePolyND;
    using cbr::PiecewisePoly;
    using cbr::Spline;
    using cbr::PiecewiseLinear;

    constexpr double epsDer = 1e-8;

    auto group_norm = [](const Tangent & t1, const Tangent & t2) {
        if constexpr (Dimension == 1) {
          return t1 - t2;
        } else {
          return (t1 - t2).norm();
        }
      };

    auto group_dist = [](const group_t & p1, const group_t & p2) {
        if constexpr (Dimension == 1) {
          return (p1.inverse() * p2).log();
        } else {
          return (p1.inverse() * p2).log().norm();
        }
      };

    auto group_tgt_zero = []() {
        if constexpr (Dimension == 1) {
          return 0.;
        } else {
          return Tangent::Zero();
        }
      };

    auto dist_to_bp = [&](const double x) {
        double d = 1e10;
        for (int i = 0; i < bp.size(); i++) {
          const double dist = abs(x - bp[i]);
          if (dist < d) {
            d = dist;
          }
        }
        return d;
      };

    const auto & bpLast = bp[bp.size() - 1];
    const auto & bpFirst = bp[0];

    // Linear fit
    auto pp = PiecewiseLinear::fitLie(bp, poses);

    for (std::size_t i = 0; i < poses.size(); i++) {
      EXPECT_NEAR(group_dist(poses[i], pp.val(bp[i])), 0., eps);
    }

    for (double x = bpFirst; x <= bpLast - 2. * epsDer; x += .1) {
      const auto derAnalytic = pp.der(x);
      const auto p0 = pp.val(x);
      const auto p1 = pp.val(x + epsDer);
      const Tangent derNumeric = (p0.inverse() * p1).log() / epsDer;
      if (dist_to_bp(x) > epsDer) {
        EXPECT_NEAR(group_norm(derAnalytic, derNumeric), 0., 1e-4);
      }
    }

    // Linear extrap
    EXPECT_NEAR(group_dist(poses.back(), pp.val(bpLast + 1.)), 0., eps);
    EXPECT_NEAR(group_dist(poses.back(), pp.val(bpLast + 2.)), 0., eps);
    EXPECT_NEAR(group_dist(poses.front(), pp.val(bpFirst - 1.)), 0., eps);
    EXPECT_NEAR(group_dist(poses.front(), pp.val(bpFirst - 2.)), 0., eps);
    EXPECT_NEAR(group_norm(pp.der(bpLast + 1.), group_tgt_zero()), 0., eps);
    EXPECT_NEAR(group_norm(pp.der(bpLast + 2.), group_tgt_zero()), 0., eps);
    EXPECT_NEAR(group_norm(pp.der(bpFirst - 1.), group_tgt_zero()), 0., eps);
    EXPECT_NEAR(group_norm(pp.der(bpFirst - 2.), group_tgt_zero()), 0., eps);

    pp.set_extrap(PiecewisePoly::EXTRAP::LINEAR);

    auto omegaLast = (poses[3].inverse() * poses[4]).log();
    auto omegaFirst = (poses[0].inverse() * poses[1]).log();

    EXPECT_NEAR(group_dist(poses.back() * group_t::exp(omegaLast), pp.val(bpLast + 1.)), 0., eps);
    EXPECT_NEAR(
      group_dist(poses.back() * group_t::exp(2. * omegaLast), pp.val(bpLast + 2.)), 0.,
      eps);
    EXPECT_NEAR(
      group_dist(poses.front() * group_t::exp(-omegaFirst), pp.val(bpFirst - 1.)), 0.,
      eps);
    EXPECT_NEAR(
      group_dist(poses.front() * group_t::exp(-2. * omegaFirst), pp.val(bpFirst - 2.)),
      0.,
      eps);
    EXPECT_NEAR(group_norm(pp.der(bpLast + 1.), omegaLast), 0., eps);
    EXPECT_NEAR(group_norm(pp.der(bpLast + 2.), omegaLast), 0., eps);
    EXPECT_NEAR(group_norm(pp.der(bpFirst - 1.), omegaFirst), 0., eps);
    EXPECT_NEAR(group_norm(pp.der(bpFirst - 2.), omegaFirst), 0., eps);

    // Polynomial fit
    auto pp2 = Spline::fitLie(bp, poses, vels);

    for (std::size_t i = 0; i < poses.size(); i++) {
      EXPECT_NEAR(group_dist(poses[i], pp2.val(bp[i])), 0., eps);
    }

    for (std::size_t i = 0; i < vels.size() - 1; i++) {
      EXPECT_NEAR(group_norm(vels[i], pp2.der(bp[i])), 0., eps);
    }

    for (double x = bpFirst; x <= bpLast - 2. * epsDer; x += .1) {
      const auto derAnalytic = pp2.der(x);
      const auto p0 = pp2.val(x);
      const auto p1 = pp2.val(x + epsDer);
      const Tangent derNumeric = (p0.inverse() * p1).log() / epsDer;
      EXPECT_NEAR(group_norm(derAnalytic, derNumeric), 0., 1e-1);
    }

    // Polynomial extrap
    EXPECT_NEAR(group_dist(poses.back(), pp2.val(bpLast + 1.)), 0., eps);
    EXPECT_NEAR(group_dist(poses.back(), pp2.val(bpLast + 2.)), 0., eps);
    EXPECT_NEAR(group_dist(poses.front(), pp2.val(bpFirst - 1.)), 0., eps);
    EXPECT_NEAR(group_dist(poses.front(), pp2.val(bpFirst - 2.)), 0., eps);
    EXPECT_NEAR(group_norm(pp2.der(bpLast + 1.), group_tgt_zero()), 0., eps);
    EXPECT_NEAR(group_norm(pp2.der(bpLast + 2.), group_tgt_zero()), 0., eps);
    EXPECT_NEAR(group_norm(pp2.der(bpFirst - 1.), group_tgt_zero()), 0., eps);
    EXPECT_NEAR(group_norm(pp2.der(bpFirst - 2.), group_tgt_zero()), 0., eps);

    pp2.set_extrap(PiecewisePoly::EXTRAP::LINEAR);

    omegaLast = vels.back();
    omegaFirst = vels.front();

    EXPECT_NEAR(group_dist(poses.back() * group_t::exp(omegaLast), pp2.val(bpLast + 1.)), 0., eps);
    EXPECT_NEAR(
      group_dist(poses.back() * group_t::exp(2. * omegaLast), pp2.val(bpLast + 2.)), 0.,
      eps);
    EXPECT_NEAR(
      group_dist(poses.front() * group_t::exp(-omegaFirst), pp2.val(bpFirst - 1.)), 0.,
      eps);
    EXPECT_NEAR(
      group_dist(poses.front() * group_t::exp(-2. * omegaFirst), pp2.val(bpFirst - 2.)),
      0.,
      eps);
    EXPECT_NEAR(group_norm(pp2.der(bpLast + 1.), omegaLast), 0., eps);
    EXPECT_NEAR(group_norm(pp2.der(bpLast + 2.), omegaLast), 0., eps);
    EXPECT_NEAR(group_norm(pp2.der(bpFirst - 1.), omegaFirst), 0., eps);
    EXPECT_NEAR(group_norm(pp2.der(bpFirst - 2.), omegaFirst), 0., eps);

    // Polynomial continuity
    for (int i = 0; i < bp.size(); i++) {
      const auto derNm1 = pp2.der(bp[i] - 1e-12);
      const auto derNp1 = pp2.der(bp[i] + 1e-12);
      EXPECT_NEAR(group_norm(derNm1, derNp1), 0., 1e-5);
    }

    // C1 Polynomial fit
    auto pp3 = Spline::fitLie(bp, poses);

    for (std::size_t i = 0; i < poses.size() - 1; i++) {
      EXPECT_NEAR(group_dist(poses[i], pp3.val(bp[i] - eps)), 0., 1e-8);
      EXPECT_NEAR(group_dist(poses[i], pp3.val(bp[i] + eps)), 0., 1e-8);
    }

    // C1 Polynomial extrap
    EXPECT_NEAR(group_dist(poses.back(), pp3.val(bpLast + 1.)), 0., eps);
    EXPECT_NEAR(group_dist(poses.back(), pp3.val(bpLast + 2.)), 0., eps);
    EXPECT_NEAR(group_dist(poses.front(), pp3.val(bpFirst - 1.)), 0., eps);
    EXPECT_NEAR(group_dist(poses.front(), pp3.val(bpFirst - 2.)), 0., eps);
    EXPECT_NEAR(group_norm(pp3.der(bpLast + 1.), group_tgt_zero()), 0., eps);
    EXPECT_NEAR(group_norm(pp3.der(bpLast + 2.), group_tgt_zero()), 0., eps);
    EXPECT_NEAR(group_norm(pp3.der(bpFirst - 1.), group_tgt_zero()), 0., eps);
    EXPECT_NEAR(group_norm(pp3.der(bpFirst - 2.), group_tgt_zero()), 0., eps);

    // C1 Polynomial continuity
    pp3.set_extrap(PiecewisePoly::EXTRAP::LINEAR);

    for (int i = 0; i < bp.size() - 1; i++) {
      const auto derNm1 = pp3.der(bp[i] - 1e-12);
      const auto derNp1 = pp3.der(bp[i] + 1e-12);
      EXPECT_NEAR(group_norm(derNm1, derNp1), 0., 1e-5);
    }
  }

  cbr::PiecewisePolyND::row_t bp;
  cbr::vector_aligned<group_t> poses;
  cbr::vector_aligned<Tangent> vels;
};

TEST(PiecewisePoly, LieGroupSO2)
{
  LieGroupTest<Sophus::SO2d> testSO2;

  testSO2.bp.resize(1, 5);
  testSO2.bp << 0., 1., 3., 6., 7.;

  testSO2.poses.push_back(Sophus::SO2d{});
  testSO2.poses.push_back(Sophus::SO2d(0.));
  testSO2.poses.push_back(Sophus::SO2d(M_PI / 4.));
  testSO2.poses.push_back(Sophus::SO2d(M_PI / 2.));
  testSO2.poses.push_back(Sophus::SO2d(0.));

  testSO2.vels.push_back(0.);
  testSO2.vels.push_back(1.);
  testSO2.vels.push_back(1.);
  testSO2.vels.push_back(2.);
  testSO2.vels.push_back(3.);

  testSO2();
}

TEST(PiecewisePoly, LieGroupSE2)
{
  LieGroupTest<Sophus::SE2d> testSE2;

  testSE2.bp.resize(1, 5);
  testSE2.bp << 0., 1., 3., 6., 7.;

  testSE2.poses.push_back(Sophus::SE2d{});
  testSE2.poses.push_back(Sophus::SE2d(0., {1., 0.}));
  testSE2.poses.push_back(Sophus::SE2d(M_PI / 4., {1., 1.}));
  testSE2.poses.push_back(Sophus::SE2d(M_PI / 4., {0., 0.}));
  testSE2.poses.push_back(Sophus::SE2d(0., {0., 0.}));

  testSE2.vels.push_back({0., 0., 0.});
  testSE2.vels.push_back({1., 1., 1.});
  testSE2.vels.push_back({1.5, 2., 3.});
  testSE2.vels.push_back({2., 2., 0.});
  testSE2.vels.push_back({0., 0., 1.});

  testSE2();
}

TEST(PiecewisePoly, LieGroupSO3)
{
  LieGroupTest<Sophus::SO3d> testSO3;

  testSO3.bp.resize(1, 5);
  testSO3.bp << 0., 1., 3., 6., 7.;

  testSO3.poses.push_back(Sophus::SO3d{});
  testSO3.poses.push_back(Sophus::SO3d(Eigen::Quaterniond(1., 0., 0., 0.)));
  testSO3.poses.push_back(Sophus::SO3d(Eigen::Quaterniond(1., 1., 0., 0.)));
  testSO3.poses.push_back(Sophus::SO3d(Eigen::Quaterniond(1., 0., 1., 0.)));
  testSO3.poses.push_back(Sophus::SO3d(Eigen::Quaterniond(1., 0., 0., 1.)));

  testSO3.vels.push_back({0., 0., 0.});
  testSO3.vels.push_back({1., 1., 1.});
  testSO3.vels.push_back({1.5, 2., 3.});
  testSO3.vels.push_back({2., 2., 0.});
  testSO3.vels.push_back({0., 1., 0.});

  testSO3();
}

TEST(PiecewisePoly, LieGroupSE3)
{
  LieGroupTest<Sophus::SE3d> testSE3;

  testSE3.bp.resize(1, 5);
  testSE3.bp << 0., 1., 3., 6., 7.;

  testSE3.poses.push_back(Sophus::SE3d{});
  testSE3.poses.push_back(Sophus::SE3d(Eigen::Quaterniond(1., 0., 0., 0.), {0., 1., 2.}));
  testSE3.poses.push_back(Sophus::SE3d(Eigen::Quaterniond(1., 1., 0., 0.), {1., 1., 5.}));
  testSE3.poses.push_back(Sophus::SE3d(Eigen::Quaterniond(1., 0., 1., 0.), {0., 2., 2.}));
  testSE3.poses.push_back(Sophus::SE3d(Eigen::Quaterniond(1., 0., 0., 1.), {2., 1., 3.}));

  testSE3.vels.push_back((Sophus::Vector6d() << 0., 0., 0., 1.5, 2., 3.).finished());
  testSE3.vels.push_back((Sophus::Vector6d() << 1., 1., 1., 1.5, 2., 3.).finished());
  testSE3.vels.push_back((Sophus::Vector6d() << 1.5, 2., 3., 1.5, 2., 3.).finished());
  testSE3.vels.push_back((Sophus::Vector6d() << 2., 2., 0., 2., 2., 0.).finished());
  testSE3.vels.push_back((Sophus::Vector6d() << 0., 0., 0., 0., 0., 0.).finished());

  testSE3();
}
