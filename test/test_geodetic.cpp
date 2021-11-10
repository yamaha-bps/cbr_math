// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE


#include <gtest/gtest.h>

#include <vector>
#include <array>
#include <tuple>
#include <utility>

#include "cbr_math/geodetic.hpp"

std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> llas{
  {0., 0., 0.},
  {0.1, 0.1, 100.},
  {0.3, 0.5, 1000.},
  {0.4, 0.7, 24.},
  {0.5, 0.2, -100.},
  {0.6, 0.1, 450.},
  {0.7, 0.9, 5000.},
  {0.8, 0.1, 300.}
};

std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> qs{
  {1., 0., 0., 0.},
  {1., 1., 0., 0.},
  {1., 1., 1., 0.},
  {1., 1., 1., 1.},
  {0., 1., 1., 1.},
  {0., 0., 1., 1.},
  {0., 0., 0., 1.},
  {1., 0., 0., 1.}
};

TEST(Geodetic, geographic2geocentric)
{
  Eigen::Vector3d lla(0., 0., 0.);
  Eigen::Vector3d llr;

  cbr::geo::geographic2geocentric(lla, llr);
  EXPECT_DOUBLE_EQ(llr[0], 0.);
  EXPECT_DOUBLE_EQ(llr[1], 0.);
  EXPECT_DOUBLE_EQ(llr[2], cbr::geo::WGS84::a);

  lla[0] = M_PI_2;
  cbr::geo::geographic2geocentric(lla, llr);
  EXPECT_DOUBLE_EQ(llr[0], M_PI_2);
  EXPECT_DOUBLE_EQ(llr[1], 0.);
  EXPECT_DOUBLE_EQ(llr[2], cbr::geo::WGS84::b);

  lla[0] = M_PI_4;
  cbr::geo::geographic2geocentric(lla, llr);
  EXPECT_LT(llr[0], M_PI_4);
  EXPECT_DOUBLE_EQ(llr[1], 0.);
  EXPECT_LT(llr[2], cbr::geo::WGS84::a);
  EXPECT_GT(llr[2], cbr::geo::WGS84::b);

  for (double longi = 0.; longi < 2. * M_PI; longi += 0.1) {
    Eigen::Vector3d llr2;
    lla[1] = longi;
    cbr::geo::geographic2geocentric(lla, llr2);
    EXPECT_DOUBLE_EQ(llr2[0], llr[0]);
    EXPECT_DOUBLE_EQ(llr2[1], longi);
    EXPECT_DOUBLE_EQ(llr2[2], llr[2]);
  }

  for (const auto & a : llas) {
    Eigen::Vector3d r1;
    cbr::geo::geographic2geocentric(a, r1);
    auto r2 = cbr::geo::geographic2geocentric(a);
    Eigen::Vector3d r3 = a;
    cbr::geo::geographic2geocentric(r3, r3);

    EXPECT_DOUBLE_EQ(r1[0], r2[0]);
    EXPECT_DOUBLE_EQ(r1[1], r2[1]);
    EXPECT_DOUBLE_EQ(r1[2], r2[2]);

    EXPECT_DOUBLE_EQ(r1[0], r3[0]);
    EXPECT_DOUBLE_EQ(r1[1], r3[1]);
    EXPECT_DOUBLE_EQ(r1[2], r3[2]);
  }
}

TEST(Geodetic, llr2ecef)
{
  for (const auto & lla : llas) {
    Eigen::Vector3d ecef, ecef2, llr, llr2;
    cbr::geo::lla2ecef(lla, ecef);
    cbr::geo::geographic2geocentric(lla, llr);
    cbr::geo::ecef2llr(ecef, llr2);
    cbr::geo::llr2ecef(llr2, ecef2);

    EXPECT_NEAR(llr[0], llr2[0], 1e-8);
    EXPECT_NEAR(llr[1], llr2[1], 1e-8);
    EXPECT_NEAR(llr[2], llr2[2], 1e-8);

    EXPECT_NEAR(ecef[0], ecef2[0], 1e-8);
    EXPECT_NEAR(ecef[1], ecef2[1], 1e-8);
    EXPECT_NEAR(ecef[2], ecef2[2], 1e-8);
  }
  for (const auto & lla : llas) {
    const Eigen::Vector3d ecef = cbr::geo::lla2ecef(lla);
    const Eigen::Vector3d llr = cbr::geo::geographic2geocentric(lla);
    const Eigen::Vector3d llr2 = cbr::geo::ecef2llr(ecef);
    const Eigen::Vector3d ecef2 = cbr::geo::llr2ecef(llr2);

    EXPECT_NEAR(llr[0], llr2[0], 1e-8);
    EXPECT_NEAR(llr[1], llr2[1], 1e-8);
    EXPECT_NEAR(llr[2], llr2[2], 1e-8);

    EXPECT_NEAR(ecef[0], ecef2[0], 1e-8);
    EXPECT_NEAR(ecef[1], ecef2[1], 1e-8);
    EXPECT_NEAR(ecef[2], ecef2[2], 1e-8);
  }
}

TEST(Geodetic, lla2ecef)
{
  for (const auto & lla : llas) {
    Eigen::Vector3d ecef, lla2;
    cbr::geo::lla2ecef(lla, ecef);
    cbr::geo::ecef2lla(ecef, lla2);

    EXPECT_NEAR(lla[0], lla2[0], 1e-8);
    EXPECT_NEAR(lla[1], lla2[1], 1e-8);
    EXPECT_NEAR(lla[2], lla2[2], 1e-8);
  }
  for (const auto & lla : llas) {
    const Eigen::Vector3d ecef = cbr::geo::lla2ecef(lla);
    const Eigen::Vector3d lla2 = cbr::geo::ecef2lla(ecef);

    EXPECT_NEAR(lla[0], lla2[0], 1e-8);
    EXPECT_NEAR(lla[1], lla2[1], 1e-8);
    EXPECT_NEAR(lla[2], lla2[2], 1e-8);
  }
}

TEST(Geodetic, geo2ecef)
{
  auto test = [](const auto & q) {
      const Eigen::Vector3d x = q.toRotationMatrix() * Eigen::Vector3d::UnitX();
      const Eigen::Vector3d y = q.toRotationMatrix() * Eigen::Vector3d::UnitY();
      const Eigen::Vector3d z = q.toRotationMatrix() * Eigen::Vector3d::UnitZ();

      EXPECT_NEAR(x[0], 0., 1e-12);
      EXPECT_NEAR(x[1], 0., 1e-12);
      EXPECT_NEAR(x[2], 1., 1e-12);

      EXPECT_NEAR(y[0], 0., 1e-12);
      EXPECT_NEAR(y[1], -1., 1e-12);
      EXPECT_NEAR(y[2], 0., 1e-12);

      EXPECT_NEAR(z[0], 1., 1e-12);
      EXPECT_NEAR(z[1], 0., 1e-12);
      EXPECT_NEAR(z[2], 0., 1e-12);
    };

  Eigen::Vector2d ll{0., 0.};
  {
    Eigen::Quaterniond qIn = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond qOut;
    cbr::geo::geo2ecef(qIn, ll, qOut);
    test(qOut);
  }
  {
    Eigen::Quaterniond qIn = Eigen::Quaterniond::Identity();
    Eigen::Quaterniond qOut = cbr::geo::geo2ecef(qIn, ll);
    test(qOut);
  }
}

template<int First, int Last, typename Lambda>
inline void static_for(Lambda const & f)
{
  if constexpr (First < Last)
  {
    f(std::integral_constant<int, First>{});
    static_for<First + 1, Last>(f);
  }
}

TEST(Geodetic, imu2nwu)
{
  auto tests = std::make_tuple(
    std::make_pair(
      std::vector<Eigen::Vector3d>{
    {0., 0., 0.},
    {0., M_PI_4, 0.},
    {0., -M_PI_4, 0.},
    {0., M_PI_2, 0.},
    {0., -M_PI_2, 0.},
    {0., M_PI, 0.},
    {M_PI_2, 0., 0.},
    {-M_PI_2, 0., 0.}},
      [](const auto & q) {
        EXPECT_NEAR(q.w(), 1., 1e-12);
        EXPECT_NEAR(q.x(), 0., 1e-12);
        EXPECT_NEAR(q.y(), 0., 1e-12);
        EXPECT_NEAR(q.z(), 0., 1e-12);
      }
    ),
    std::make_pair(
      std::vector<Eigen::Vector3d>{
    {M_PI_4, 0., 0.},
    {M_PI_4, M_PI_4, 0.},
    {M_PI_4, -M_PI_4, 0.},
    {M_PI_4, M_PI_2, 0.},
    {M_PI_4, -M_PI_2, 0.},
    {M_PI_4, M_PI, 0.}},
      [](const auto & q) {
        EXPECT_NEAR(q.x(), 0., 1e-12);
        EXPECT_LE(q.y(), 0.);
        EXPECT_NEAR(q.z(), 0., 1e-12);
      }
    ),
    std::make_pair(
      std::vector<Eigen::Vector3d>{
    {-M_PI_4, 0., 0.},
    {-M_PI_4, M_PI_4, 0.},
    {-M_PI_4, -M_PI_4, 0.},
    {-M_PI_4, M_PI_2, 0.},
    {-M_PI_4, -M_PI_2, 0.},
    {-M_PI_4, M_PI, 0.}},
      [](const auto & q) {
        EXPECT_NEAR(q.x(), 0., 1e-12);
        EXPECT_GE(q.y(), 0.);
        EXPECT_NEAR(q.z(), 0., 1e-12);
      }
    )
  );

  static_for<0, std::tuple_size_v<decltype(tests)>>(
    [&](auto i) {
      const auto & test = std::get<i.value>(tests);
      for (const auto & lla  : test.first) {
        {
          Eigen::Quaterniond qIn = Eigen::Quaterniond::Identity();
          Eigen::Quaterniond qOut, qIn2;
          cbr::geo::imu2nwu(qIn, lla, qOut);
          test.second(qOut);
          cbr::geo::nwu2imu(qOut, lla, qIn2);
          EXPECT_TRUE(qIn2.isApprox(qIn, 1e-12));
        }
        {
          Eigen::Quaterniond qIn = Eigen::Quaterniond::Identity();
          Eigen::Quaterniond qOut = cbr::geo::imu2nwu(qIn, lla);
          Eigen::Quaterniond qIn2 = cbr::geo::nwu2imu(qOut, lla);
          test.second(qOut);
          EXPECT_TRUE(qIn2.isApprox(qIn, 1e-12));
        }
      }
    });
}

TEST(Geodetic, lla2gnomonic)
{
  constexpr double eps = 1e-12;

  using cbr::geo::lla2gnomonic;
  using cbr::geo::gnomonic2lla;

  Eigen::Vector3d llaRef{0., 0., 0.};
  Eigen::Vector3d lla{0., M_PI, 0.};

  auto [success0, xyz0] = lla2gnomonic(lla, llaRef);
  EXPECT_FALSE(success0);

  lla.y() = M_PI_2;
  auto [success1, xyz1] = lla2gnomonic(lla, llaRef);
  EXPECT_FALSE(success1);

  lla.y() = 0.;
  auto [success2, xyz2] = lla2gnomonic(lla, llaRef);
  EXPECT_TRUE(success2);
  EXPECT_NEAR(xyz2[0], 0., eps);
  EXPECT_NEAR(xyz2[1], 0., eps);
  EXPECT_NEAR(xyz2[2], 0., eps);

  lla.x() = M_PI_4;
  lla.z() = 10.;
  auto [success3, xyz3] = lla2gnomonic(lla, llaRef);
  EXPECT_TRUE(success3);
  EXPECT_GE(xyz3[0], 0.);
  EXPECT_NEAR(xyz3[1], 0., eps);
  EXPECT_NEAR(xyz3[2], 10., eps);

  lla.x() = -M_PI_4;
  lla.y() = M_PI_4;
  llaRef.y() = M_PI_4;
  lla.z() = 10.;
  auto [success4, xyz4] = lla2gnomonic(lla, llaRef);
  EXPECT_TRUE(success4);
  EXPECT_LE(xyz4[0], 0.);
  EXPECT_NEAR(xyz4[1], 0., eps);
  EXPECT_NEAR(xyz4[2], 10., eps);

  lla.x() = -M_PI_4;
  lla.y() = M_PI_4;
  llaRef.y() = 0.;
  lla.z() = 20.;
  auto [success5, xyz5] = lla2gnomonic(lla, llaRef);
  EXPECT_TRUE(success5);
  EXPECT_LE(xyz5[0], 0.);
  EXPECT_LE(xyz5[1], 0.);
  EXPECT_NEAR(xyz5[2], 20., eps);

  for (std::size_t i = 0; i < llas.size(); i++) {
    for (std::size_t j = 0; j < llas.size(); j++) {
      const auto & llaTmp = llas[i];
      const auto & llaRefTmp = llas[j];
      Eigen::Vector3d xyz, lla2;

      if (lla2gnomonic(llaTmp, llaRefTmp, xyz)) {
        gnomonic2lla(xyz, llaRefTmp, lla2);

        EXPECT_NEAR(llaTmp[0], lla2[0], 1e-8);
        EXPECT_NEAR(llaTmp[1], lla2[1], 1e-8);
        EXPECT_NEAR(llaTmp[2], lla2[2], 1e-8);
      }
    }
  }
}

TEST(Geodetic, lla2nwu)
{
  for (std::size_t i = 0; i < llas.size(); i++) {
    for (std::size_t j = 0; j < llas.size(); j++) {
      const auto & lla = llas[i];
      const auto & llaRef = llas[j];
      // const auto & q = qs[i];
      Eigen::Vector3d nwu, lla2;

      cbr::geo::lla2nwu(lla, llaRef, nwu);
      cbr::geo::nwu2lla(nwu, llaRef, lla2);

      EXPECT_NEAR(lla[0], lla2[0], 1e-8);
      EXPECT_NEAR(lla[1], lla2[1], 1e-8);
      EXPECT_NEAR(lla[2], lla2[2], 1e-8);
    }
  }
}


TEST(Geodetic, lla2nwu_flip)
{
  // two equator points opposite each other
  Eigen::Vector3d lla{0, -M_PI / 2, 0};
  Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();

  Eigen::Vector3d llaFrame{0, M_PI / 2, 0};
  Eigen::Quaterniond quatFrame = Eigen::Quaterniond::Identity();

  Eigen::Vector3d xyzInFrame;
  Eigen::Quaterniond quatInFrame;

  // express one point in the frame of the other
  // - position should be [0, 0, -2*r]
  // - orientation amounts to a π-rotation around the x axis
  cbr::geo::lla2nwu(
    lla, quat,
    llaFrame, quatFrame,
    xyzInFrame, quatInFrame
  );

  EXPECT_NEAR(xyzInFrame.x(), 0, 1e-8);
  EXPECT_NEAR(xyzInFrame.y(), 0, 1e-8);
  EXPECT_NEAR(xyzInFrame.z(), -2 * cbr::geo::WGS84::a, 1e-8);

  EXPECT_NEAR(quatInFrame.w(), 0, 1e-8);
  EXPECT_NEAR(std::abs(quatInFrame.x()), 1, 1e-8);
  EXPECT_NEAR(quatInFrame.y(), 0, 1e-8);
  EXPECT_NEAR(quatInFrame.z(), 0, 1e-8);
}


TEST(Geodetic, nwu2lla_rot)
{
  // at equator pointing north
  Eigen::Vector3d llaFrame{0, 0, 0};
  Eigen::Quaterniond quatFrame = Eigen::Quaterniond::Identity();

  // translate 45 degrees
  Eigen::Vector3d xyzInFrame{0, cbr::geo::WGS84::a, -cbr::geo::WGS84::a};
  Eigen::Quaterniond quatInFrame = Eigen::Quaterniond::Identity();

  Eigen::Vector3d lla;
  Eigen::Quaterniond quat;
  cbr::geo::nwu2lla(xyzInFrame, quatInFrame, llaFrame, quatFrame, lla, quat);

  // resulting lla should be rotated 90 deg
  // orientation should be rotated 90 deg around x axis
  ASSERT_NEAR(lla.x(), 0, 1e-10);
  ASSERT_NEAR(lla.y(), -M_PI_2, 1e-10);
  ASSERT_NEAR(lla.z(), 0, 1e-10);

  ASSERT_NEAR(quat.w(), M_SQRT1_2, 1e-10);
  ASSERT_NEAR(quat.x(), M_SQRT1_2, 1e-10);
  ASSERT_NEAR(quat.y(), 0, 1e-10);
  ASSERT_NEAR(quat.z(), 0, 1e-10);
}

TEST(Geodetic, ecef2lla_pose)
{
  {
    // at equator x 0
    Eigen::Vector3d lla(0, 0, 5);
    Eigen::Quaterniond q_nwu = Eigen::Quaterniond::Identity();

    auto [t_ecef, q_ecef] = cbr::geo::lla2ecef(lla, q_nwu);

    Eigen::Matrix3d R_test = (Eigen::Matrix3d() << 0, 0, 1, 0, -1, 0, 1, 0, 0).finished();
    Eigen::Quaterniond q_test(R_test);

    ASSERT_TRUE(t_ecef.isApprox(Eigen::Vector3d(cbr::geo::WGS84::a + 5, 0, 0)));
    ASSERT_TRUE(q_ecef.isApprox(q_test));

    auto [lla_copy, q_nwu_copy] = cbr::geo::ecef2lla(t_ecef, q_ecef);

    if (q_nwu.w() < 0) {q_nwu.coeffs() *= -1;}
    if (q_nwu_copy.w() < 0) {q_nwu_copy.coeffs() *= -1;}

    ASSERT_TRUE(lla_copy.isApprox(lla));
    ASSERT_TRUE(q_nwu_copy.isApprox(q_nwu));
  }

  {
    // at equator 0 y
    Eigen::Vector3d lla(0, M_PI_2, 5);
    Eigen::Quaterniond q_nwu = Eigen::Quaterniond::Identity();

    auto [t_ecef, q_ecef] = cbr::geo::lla2ecef(lla, q_nwu);

    Eigen::Matrix3d R_test = (Eigen::Matrix3d() << 0, 1, 0, 0, 0, 1, 1, 0, 0).finished();
    Eigen::Quaterniond q_test(R_test);

    ASSERT_TRUE(t_ecef.isApprox(Eigen::Vector3d(0, cbr::geo::WGS84::a + 5, 0)));
    ASSERT_TRUE(q_ecef.isApprox(q_test));

    auto [lla_copy, q_nwu_copy] = cbr::geo::ecef2lla(t_ecef, q_ecef);

    if (q_nwu.w() < 0) {q_nwu.coeffs() *= -1;}
    if (q_nwu_copy.w() < 0) {q_nwu_copy.coeffs() *= -1;}

    ASSERT_TRUE(lla_copy.isApprox(lla));
    ASSERT_TRUE(q_nwu_copy.isApprox(q_nwu));
  }

  // random llas
  srand(5);
  for (auto i = 0u; i < 10u; ++i) {
    Eigen::Vector3d lla = Eigen::Vector3d::Random();
    Eigen::Quaterniond q_nwu = Eigen::Quaterniond::UnitRandom();

    auto [t_ecef, q_ecef] = cbr::geo::lla2ecef(lla, q_nwu);
    auto [lla_copy, q_nwu_copy] = cbr::geo::ecef2lla(t_ecef, q_ecef);
    auto [t_ecef_copy, q_ecef_copy] = cbr::geo::lla2ecef(lla_copy, q_nwu_copy);

    if (q_nwu.w() < 0) {q_nwu.coeffs() *= -1;}
    if (q_nwu_copy.w() < 0) {q_nwu_copy.coeffs() *= -1;}
    if (q_ecef.w() < 0) {q_ecef.coeffs() *= -1;}
    if (q_ecef_copy.w() < 0) {q_ecef_copy.coeffs() *= -1;}

    ASSERT_TRUE(lla_copy.isApprox(lla, 1e-8));
    ASSERT_TRUE(q_nwu_copy.isApprox(q_nwu));
    ASSERT_TRUE(t_ecef_copy.isApprox(t_ecef));
    ASSERT_TRUE(q_ecef_copy.isApprox(q_ecef));
  }
}
