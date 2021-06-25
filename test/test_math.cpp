// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE


#include <gtest/gtest.h>

#include <vector>
#include <array>
#include <utility>

#include "cbr_math/math.hpp"
#include "cbr_math/eigen_traits.hpp"

TEST(Math, wrap2Pi)
{
  EXPECT_DOUBLE_EQ(cbr::wrap2Pi(0.), 0.);
  EXPECT_DOUBLE_EQ(cbr::wrap2Pi(M_PI_4), M_PI_4);
  EXPECT_DOUBLE_EQ(cbr::wrap2Pi(M_PI_2), M_PI_2);
  EXPECT_DOUBLE_EQ(cbr::wrap2Pi(M_PI), M_PI);
  EXPECT_DOUBLE_EQ(cbr::wrap2Pi(5. * M_PI_2), M_PI_2);
  EXPECT_DOUBLE_EQ(cbr::wrap2Pi(-M_PI_4), 7. * M_PI_4);
  EXPECT_DOUBLE_EQ(cbr::wrap2Pi(-M_PI_2), 3. * M_PI_2);
  EXPECT_DOUBLE_EQ(cbr::wrap2Pi(-M_PI), M_PI);
}

TEST(Math, wrapPi)
{
  EXPECT_NEAR(cbr::wrapPi(-M_PI + 1e-12), -M_PI, 1e-10);
  EXPECT_DOUBLE_EQ(cbr::wrapPi(-M_PI_2), -M_PI_2);
  EXPECT_DOUBLE_EQ(cbr::wrapPi(-M_PI_4), -M_PI_4);
  EXPECT_DOUBLE_EQ(cbr::wrapPi(0.), 0.);
  EXPECT_DOUBLE_EQ(cbr::wrapPi(M_PI_4), M_PI_4);
  EXPECT_DOUBLE_EQ(cbr::wrapPi(M_PI_2), M_PI_2);
  EXPECT_NEAR(cbr::wrapPi(M_PI - 1e-12), M_PI, 1e-10);
  EXPECT_DOUBLE_EQ(cbr::wrapPi(2. * M_PI), 0.);
  EXPECT_DOUBLE_EQ(cbr::wrapPi(3. * M_PI_2), -M_PI_2);
}


TEST(Math, factorial)
{
  constexpr auto v0 = cbr::factorial<uint64_t>(0);
  constexpr auto v1 = cbr::factorial<uint64_t>(1);
  constexpr auto v2 = cbr::factorial<uint64_t>(2);
  constexpr auto v3 = cbr::factorial<uint64_t>(3);
  constexpr auto v4 = cbr::factorial<uint64_t>(4);
  constexpr auto v5 = cbr::factorial<uint64_t>(5);
  constexpr auto v6 = cbr::factorial<uint64_t>(6);
  constexpr auto v7 = cbr::factorial<uint64_t>(7);

  ASSERT_EQ(v0, 1LU);
  ASSERT_EQ(v1, 1LU);
  ASSERT_EQ(v2, 2LU);
  ASSERT_EQ(v3, 6LU);
  ASSERT_EQ(v4, 24LU);
  ASSERT_EQ(v5, 120LU);
  ASSERT_EQ(v6, 720LU);
  ASSERT_EQ(v7, 5040LU);

  constexpr auto v30 = cbr::factorial<uint64_t>(0, 3);
  constexpr auto v31 = cbr::factorial<uint64_t>(1, 3);
  constexpr auto v32 = cbr::factorial<uint64_t>(2, 3);
  constexpr auto v33 = cbr::factorial<uint64_t>(3, 3);
  constexpr auto v34 = cbr::factorial<uint64_t>(4, 3);
  constexpr auto v35 = cbr::factorial<uint64_t>(5, 3);
  constexpr auto v36 = cbr::factorial<uint64_t>(6, 3);
  constexpr auto v37 = cbr::factorial<uint64_t>(7, 3);

  ASSERT_EQ(v30, 1LU);
  ASSERT_EQ(v31, 1LU);
  ASSERT_EQ(v32, 1LU);
  ASSERT_EQ(v33, 3LU);
  ASSERT_EQ(v34, 12LU);
  ASSERT_EQ(v35, 60LU);
  ASSERT_EQ(v36, 360LU);
  ASSERT_EQ(v37, 2520LU);

  constexpr auto v40 = cbr::factorial<uint64_t>(0, 4);
  constexpr auto v41 = cbr::factorial<uint64_t>(1, 4);
  constexpr auto v42 = cbr::factorial<uint64_t>(2, 4);
  constexpr auto v43 = cbr::factorial<uint64_t>(3, 4);
  constexpr auto v44 = cbr::factorial<uint64_t>(4, 4);
  constexpr auto v45 = cbr::factorial<uint64_t>(5, 4);
  constexpr auto v46 = cbr::factorial<uint64_t>(6, 4);
  constexpr auto v47 = cbr::factorial<uint64_t>(7, 4);

  ASSERT_EQ(v40, 1LU);
  ASSERT_EQ(v41, 1LU);
  ASSERT_EQ(v42, 1LU);
  ASSERT_EQ(v43, 1LU);
  ASSERT_EQ(v44, 4LU);
  ASSERT_EQ(v45, 20LU);
  ASSERT_EQ(v46, 120LU);
  ASSERT_EQ(v47, 840LU);
}

TEST(Math, powFast)
{
  constexpr auto v10 = cbr::powFast(1., 0LU);
  constexpr auto v20 = cbr::powFast(2., 0LU);
  constexpr auto v30 = cbr::powFast(3., 0LU);
  constexpr auto v40 = cbr::powFast(4., 0LU);
  constexpr auto v50 = cbr::powFast(5., 0LU);

  ASSERT_DOUBLE_EQ(v10, 1.);
  ASSERT_DOUBLE_EQ(v20, 1.);
  ASSERT_DOUBLE_EQ(v30, 1.);
  ASSERT_DOUBLE_EQ(v40, 1.);
  ASSERT_DOUBLE_EQ(v50, 1.);

  constexpr auto v11 = cbr::powFast(1., 1LU);
  constexpr auto v21 = cbr::powFast(2., 1LU);
  constexpr auto v31 = cbr::powFast(3., 1LU);
  constexpr auto v41 = cbr::powFast(4., 1LU);
  constexpr auto v51 = cbr::powFast(5., 1LU);

  ASSERT_DOUBLE_EQ(v11, 1.);
  ASSERT_DOUBLE_EQ(v21, 2.);
  ASSERT_DOUBLE_EQ(v31, 3.);
  ASSERT_DOUBLE_EQ(v41, 4.);
  ASSERT_DOUBLE_EQ(v51, 5.);

  constexpr auto v12 = cbr::powFast(1., 2LU);
  constexpr auto v22 = cbr::powFast(2., 2LU);
  constexpr auto v32 = cbr::powFast(3., 2LU);
  constexpr auto v42 = cbr::powFast(4., 2LU);
  constexpr auto v52 = cbr::powFast(5., 2LU);

  ASSERT_DOUBLE_EQ(v12, 1.);
  ASSERT_DOUBLE_EQ(v22, 4.);
  ASSERT_DOUBLE_EQ(v32, 9.);
  ASSERT_DOUBLE_EQ(v42, 16.);
  ASSERT_DOUBLE_EQ(v52, 25.);

  constexpr auto v13 = cbr::powFast(1., 3LU);
  constexpr auto v23 = cbr::powFast(2., 3LU);
  constexpr auto v33 = cbr::powFast(3., 3LU);
  constexpr auto v43 = cbr::powFast(4., 3LU);
  constexpr auto v53 = cbr::powFast(5., 3LU);

  ASSERT_DOUBLE_EQ(v13, 1.);
  ASSERT_DOUBLE_EQ(v23, 8.);
  ASSERT_DOUBLE_EQ(v33, 27.);
  ASSERT_DOUBLE_EQ(v43, 64.);
  ASSERT_DOUBLE_EQ(v53, 125.);

  constexpr auto v53_s = cbr::powFast<3>(5.);
  constexpr auto v53_sf = cbr::powFast<3>(5.f);
  constexpr auto v53_s_u64 = cbr::powFast<3>(uint64_t(5));
  constexpr auto v53_s_i64 = cbr::powFast<3>(int64_t(5));
  constexpr auto v53_s_u32 = cbr::powFast<3>(uint32_t(5));
  constexpr auto v53_s_i32 = cbr::powFast<3>(int32_t(5));
  constexpr auto v53_s_u16 = cbr::powFast<3>(uint16_t(5));
  constexpr auto v53_s_i16 = cbr::powFast<3>(int16_t(5));
  constexpr auto v53_s_u8 = cbr::powFast<3>(uint8_t(5));
  constexpr auto v53_s_i8 = cbr::powFast<3>(int8_t(5));
  ASSERT_DOUBLE_EQ(v53_s, 125.);
  ASSERT_FLOAT_EQ(v53_sf, 125.f);
  ASSERT_EQ(v53_s_u64, uint64_t(125));
  ASSERT_EQ(v53_s_i64, int64_t(125));
  ASSERT_EQ(v53_s_u32, uint32_t(125));
  ASSERT_EQ(v53_s_i32, int32_t(125));
  ASSERT_EQ(v53_s_u16, uint16_t(125));
  ASSERT_EQ(v53_s_i16, int16_t(125));
  ASSERT_EQ(v53_s_u8, uint8_t(125));
  ASSERT_EQ(v53_s_i8, int8_t(125));
}

TEST(Math, quat2eulZYX)
{
  std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> quats;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> euls;

  quats.emplace_back(1., 0., 0., 0.);
  euls.emplace_back(0., 0., 0.);

  quats.emplace_back(0.5, 0., 0., 0.5);
  euls.emplace_back(0., 0., M_PI_2);

  quats.emplace_back(1., 0., 0., 1.);
  euls.emplace_back(0., 0., M_PI_2);

  quats.emplace_back(1., 0., 1., 0.);
  euls.emplace_back(0., M_PI_2, 0.);

  quats.emplace_back(1., 1., 0., 0.);
  euls.emplace_back(M_PI_2, 0., 0.);

  quats.emplace_back(1., 1., 0., 1.);
  euls.emplace_back(1.107148717794090, -0.729727656226967, 1.107148717794090);

  quats.emplace_back(1., 1., 1., 0.);
  euls.emplace_back(2.034443935795703, 0.729727656226967, 1.107148717794090);

  quats.emplace_back(1., 1., 1., 1.);
  euls.emplace_back(M_PI_2, 0., M_PI_2);

  for (uint64_t i = 0; i < quats.size(); i++) {
    auto eul = cbr::quat2eulZYX(quats[i]);
    ASSERT_NEAR(eul[0], euls[i][0], 1e-7);
    ASSERT_NEAR(eul[1], euls[i][1], 1e-7);
    ASSERT_NEAR(eul[2], euls[i][2], 1e-7);
  }
}

TEST(Math, eul2quatZYX)
{
  std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> quats;
  quats.emplace_back(1., 0., 0., 0.);
  quats.emplace_back(1., 0., 0., 1.);
  quats.emplace_back(1., 0., 1., 0.);
  quats.emplace_back(1., 1., 0., 0.);
  quats.emplace_back(1., 1., 0., 1.);
  quats.emplace_back(1., 1., 1., 0.);
  quats.emplace_back(1., 1., 1., 1.);

  for (auto & q : quats) {
    auto eul = cbr::quat2eulZYX(q);
    q.normalize();
    auto q2 = cbr::eul2quatZYX(eul);
    ASSERT_NEAR(q.w(), q2.w(), 1e-7);
    ASSERT_NEAR(q.x(), q2.x(), 1e-7);
    ASSERT_NEAR(q.y(), q2.y(), 1e-7);
    ASSERT_NEAR(q.z(), q2.z(), 1e-7);
  }
}

TEST(Math, quat2yaw)
{
  std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> quats;
  quats.emplace_back(1., 0., 0., 0.);
  quats.emplace_back(1., 0., 0., 1.);
  quats.emplace_back(1., 0., 1., 0.);
  quats.emplace_back(1., 1., 0., 0.);
  quats.emplace_back(1., 1., 0., 1.);
  quats.emplace_back(1., 1., 1., 0.);
  quats.emplace_back(1., 1., 1., 1.);

  for (auto & q : quats) {
    auto y = cbr::quat2yaw(q);
    auto q2 = cbr::yaw2quat(y);
    auto y2 = cbr::quat2yaw(q2);
    ASSERT_NEAR(y, y2, 1e-7);
  }
}

template<uint64_t _dim>
struct TestData
{
  std::array<uint64_t, _dim> sz;
  std::vector<uint64_t> idx;
};

TEST(Math, sub2ind)
{
  // Check constexpr
  constexpr std::array<uint64_t, 2> sz0{2, 2};
  constexpr auto sub0 = cbr::ind2sub(sz0, 2LU);

  ASSERT_EQ(sub0[0], 0LU);
  ASSERT_EQ(sub0[1], 1LU);

  // Dimension 2
  std::vector<TestData<2>> dim2(cbr::powFast<2>(4));

  uint64_t dim2_i = 1;
  uint64_t dim2_j = 1;
  for (auto & data : dim2) {
    data.sz[0] = dim2_i;
    data.sz[1] = dim2_j;
    auto numel = dim2_i * dim2_j;
    data.idx.reserve(numel);
    for (uint64_t i = 0; i < numel; i++) {
      data.idx.emplace_back(i);
    }
    dim2_i++;
    if (dim2_i == 4) {
      dim2_i = 1;
      dim2_j++;
    }
  }

  for (const auto & data : dim2) {
    const auto & sz = data.sz;
    for (const auto & idx : data.idx) {
      auto sub = cbr::ind2sub(sz, idx);
      auto idx2 = cbr::sub2ind(sz, sub);
      ASSERT_EQ(idx, idx2);
    }
  }

  // Dimension 3
  std::vector<TestData<3>> dim3(cbr::powFast<3>(4));

  uint64_t dim3_i = 1;
  uint64_t dim3_j = 1;
  uint64_t dim3_k = 1;
  for (auto & data : dim3) {
    data.sz[0] = dim3_i;
    data.sz[1] = dim3_j;
    data.sz[2] = dim3_k;
    auto numel = dim3_i * dim3_j * dim3_k;
    data.idx.reserve(numel);
    for (uint64_t i = 0; i < numel; i++) {
      data.idx.emplace_back(i);
    }
    dim3_i++;
    if (dim3_i == 4) {
      dim3_i = 1;
      dim3_j++;
    }
    if (dim3_j == 4) {
      dim3_j = 1;
      dim3_k++;
    }
  }

  for (const auto & data : dim3) {
    const auto & sz = data.sz;
    for (const auto & idx : data.idx) {
      auto sub = cbr::ind2sub(sz, idx);
      auto idx2 = cbr::sub2ind(sz, sub);
      ASSERT_EQ(idx, idx2);
    }
  }

  // Dimension 4
  std::vector<TestData<4>> dim4(cbr::powFast<4>(4));

  uint64_t dim4_i = 1;
  uint64_t dim4_j = 1;
  uint64_t dim4_k = 1;
  uint64_t dim4_l = 1;
  for (auto & data : dim4) {
    data.sz[0] = dim4_i;
    data.sz[1] = dim4_j;
    data.sz[2] = dim4_k;
    data.sz[3] = dim4_l;
    auto numel = dim4_i * dim4_j * dim4_k * dim4_l;
    data.idx.reserve(numel);
    for (uint64_t i = 0; i < numel; i++) {
      data.idx.emplace_back(i);
    }
    dim4_i++;
    if (dim4_i == 4) {
      dim4_i = 1;
      dim4_j++;
    }
    if (dim4_j == 4) {
      dim4_j = 1;
      dim4_k++;
    }
    if (dim4_k == 4) {
      dim4_k = 1;
      dim4_l++;
    }
  }

  for (const auto & data : dim4) {
    const auto & sz = data.sz;
    for (const auto & idx : data.idx) {
      auto sub = cbr::ind2sub(sz, idx);
      auto idx2 = cbr::sub2ind(sz, sub);
      ASSERT_EQ(idx, idx2);
    }
  }
}

TEST(Math, smoothSat)
{
  double x = 0.;
  cbr::smoothSatInPlace(x, -1., 1.);
  ASSERT_DOUBLE_EQ(x, 0.);

  x = 2.;
  cbr::smoothSatInPlace(x, -1., 1.);
  ASSERT_DOUBLE_EQ(x, 1.);

  x = -2.;
  cbr::smoothSatInPlace(x, -1., 1.);
  ASSERT_DOUBLE_EQ(x, -1.);

  x = M_SQRT2 / 2.;
  cbr::smoothSatInPlace(x, -1., 1., 1.);
  ASSERT_LT(x, 1.);

  x = 3.;
  cbr::smoothSatInPlace(x, 2., 2.);
  ASSERT_DOUBLE_EQ(x, 2.);

  auto xSat = cbr::smoothSat(0., -1., 1.);
  ASSERT_DOUBLE_EQ(xSat, 0.0);

  ASSERT_ANY_THROW(cbr::smoothSatInPlace(x, 1., -1.));
  ASSERT_ANY_THROW(cbr::smoothSatInPlace(x, -1., 1., -1.));
  ASSERT_ANY_THROW(cbr::smoothSatInPlace(x, -1., 1., 1.1));
}

TEST(Math, point_in_polygon)
{
  using cbr::point_in_polygon;

  cbr::vector_aligned<Eigen::Vector2d> vertices{{0., 0.}, {0., 1.}};
  Eigen::Vector2d pt{-1.0, -1.0};
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  vertices.push_back({1., 0.});
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 0.2;
  pt.y() = 0.2;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  vertices.clear();
  vertices.push_back({0., 0.});
  vertices.push_back({1., 0.});
  vertices.push_back({1., 1.});
  vertices.push_back({2., 1.});
  vertices.push_back({2., 0.});
  vertices.push_back({3., 2.});
  vertices.push_back({0., 2.});

  pt.x() = 0.5; pt.y() = 0.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 1.5; pt.y() = 0.5;
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 2.; pt.y() = 0.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 2.001; pt.y() = 0.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 3.; pt.y() = 0.5;
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 2.1; pt.y() = 1.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 1.5; pt.y() = 1.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 1.; pt.y() = 1.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 0.5; pt.y() = 1.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 0.0; pt.y() = 1.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 0.0; pt.y() = 2.5;
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 0.0; pt.y() = 0.;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 1.0; pt.y() = 2.;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 2.5; pt.y() = 2.;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 3.0; pt.y() = 2.;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 3.5; pt.y() = 2.;
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 1.0; pt.y() = 0.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 2.0; pt.y() = 1.;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 2.0; pt.y() = 1.;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 0.5; pt.y() = 0.;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 1.; pt.y() = 0.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));
  pt.x() = 1.5; pt.y() = 1.;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 2.; pt.y() = 0.5;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 2.; pt.y() = 2.;
  EXPECT_TRUE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 0.5; pt.y() = -0.5;
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 1.5; pt.y() = 0.5;
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = 1.5; pt.y() = 2.5;
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = -0.5; pt.y() = 0.;
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = -0.5; pt.y() = 1.;
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));

  pt.x() = -0.5; pt.y() = 2.;
  EXPECT_FALSE(point_in_polygon(vertices.cbegin(), vertices.cend(), pt));
}

TEST(Math, segments_intersect)
{
  using cbr::segments_intersect;

  struct Segment
  {
    Eigen::Vector2d first{0., 0.};
    Eigen::Vector2d second{0., 0.};
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  constexpr bool a = segments_intersect(
    std::array{0., 0.}, std::array{1., 0.},
    std::array{0.5, -1.}, std::array{0.5, 1.});

  static_assert(a);

  Segment seg1, seg2;

  ASSERT_ANY_THROW(segments_intersect(seg1, seg2));

  seg1.first = {0., 0.};
  seg1.second = {1., 0.};
  ASSERT_ANY_THROW(segments_intersect(seg1, seg2));

  seg2.first = {0.5, -1.};
  seg2.second = {0.5, 1.};
  bool res = false;
  ASSERT_NO_THROW(res = segments_intersect(seg1, seg2));
  EXPECT_TRUE(res);

  seg2.first = {1., -1.};
  seg2.second = {1., 1.};
  EXPECT_TRUE(segments_intersect(seg1, seg2));

  seg2.first = {0., -1.};
  seg2.second = {0., 1.};
  EXPECT_TRUE(segments_intersect(seg1, seg2));

  seg2.first = {-1., -1.};
  seg2.second = {-1., 1.};
  EXPECT_FALSE(segments_intersect(seg1, seg2));

  seg2.first = {2., -1.};
  seg2.second = {2., 1.};
  EXPECT_FALSE(segments_intersect(seg1, seg2));

  seg1.first = {1., 0.};
  seg1.second = {2., 0.};
  seg2.first = {0.5, -1.};
  seg2.second = {0.5, 1.};
  EXPECT_FALSE(segments_intersect(seg1, seg2));

  seg1.first = {0., 0.};
  seg1.second = {0.1, 0.};
  seg2.first = {0.5, -1.};
  seg2.second = {0.5, 1.};
  EXPECT_FALSE(segments_intersect(seg1, seg2));

  seg1.first = {0., 0.};
  seg1.second = {1., 5.};
  seg2.first = {0.5, -1.};
  seg2.second = {0.5, 1.};
  EXPECT_FALSE(segments_intersect(seg1, seg2));

  seg1.first = {0., 0.};
  seg1.second = {1., 0.};
  seg2 = seg1;
  EXPECT_FALSE(segments_intersect(seg1, seg2));

  seg2.first = {0., 1.};
  seg2.second = {1., 1.};
  EXPECT_FALSE(segments_intersect(seg1, seg2));
}

TEST(Math, segment_in_polygon)
{
  using cbr::segment_in_polygon;

  cbr::vector_aligned<Eigen::Vector2d> vertices;
  vertices.push_back({0., 0.});
  vertices.push_back({1., 0.});
  vertices.push_back({1., 1.});
  vertices.push_back({2., 1.});
  vertices.push_back({2., 0.});
  vertices.push_back({3., 2.});
  vertices.push_back({0., 2.});

  struct Segment
  {
    Eigen::Vector2d first{0., 0.};
    Eigen::Vector2d second{0., 0.};
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  Segment seg;

  seg.first = {0.1, 0.1};
  seg.second = {0.5, 0.5};
  EXPECT_TRUE(segment_in_polygon(vertices.begin(), vertices.end(), seg));

  seg.first = {0.1, 0.2};
  seg.second = {1.9, 1.9};
  EXPECT_TRUE(segment_in_polygon(vertices.begin(), vertices.end(), seg));

  seg.first = {0.1, 1.5};
  seg.second = {2.5, 1.5};
  EXPECT_TRUE(segment_in_polygon(vertices.begin(), vertices.end(), seg));

  seg.first = {2.1, 1.9};
  seg.second = {2.1, 0.3};
  EXPECT_TRUE(segment_in_polygon(vertices.begin(), vertices.end(), seg));

  seg.first = {2.1, 0.3};
  seg.second = {2.9, 1.99};
  EXPECT_TRUE(segment_in_polygon(vertices.begin(), vertices.end(), seg));

  seg.first = {.75, .5};
  seg.second = {1.5, 1.25};
  EXPECT_FALSE(segment_in_polygon(vertices.begin(), vertices.end(), seg));

  seg.first = {.5, .5};
  seg.second = {1.5, .5};
  EXPECT_FALSE(segment_in_polygon(vertices.begin(), vertices.end(), seg));

  seg.first = {.5, .5};
  seg.second = {2.01, .5};
  EXPECT_FALSE(segment_in_polygon(vertices.begin(), vertices.end(), seg));

  seg.second = {.5, -.5};
  EXPECT_FALSE(segment_in_polygon(vertices.begin(), vertices.end(), seg));

  seg.second = {-.5, .5};
  EXPECT_FALSE(segment_in_polygon(vertices.begin(), vertices.end(), seg));
}

TEST(SampleCovariance, StDev)
{
  {
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> data{
      Eigen::Vector3d{0, 0, 1}, Eigen::Vector3d{0, 0, 2}
    };

    auto error_function = [](const Eigen::Vector3d & sample) {
        return sample - 1.5 * Eigen::Vector3d::UnitZ();
      };

    auto cov_mat = cbr::sample_covariance(data, error_function);

    Eigen::Vector3d ez = Eigen::Vector3d::UnitZ();
    ASSERT_LE((0.5 * ez * ez.transpose() - cov_mat).norm(), 1e-9);
  }

  {
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> data{
      Eigen::Vector2d{5, 2}, Eigen::Vector2d{12, 8}, Eigen::Vector2d{18, 18},
      Eigen::Vector2d{23, 20}, Eigen::Vector2d{45, 28}
    };

    auto calc = cbr::sample_covariance(data, [](auto el) {return el;});

    Eigen::Matrix2d ans = (Eigen::Matrix2d() << 231.3, 146.1, 146.1, 105.2).finished();
    ASSERT_LE((calc - ans).norm(), 1e-9);
  }
}
