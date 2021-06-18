// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE


#include <gtest/gtest.h>

#include <vector>

#include "cbr_math/sophus.hpp"

TEST(Math, SE2_SE3)
{
  std::vector<Sophus::SE2d, Eigen::aligned_allocator<Sophus::SE2d>> poses;
  poses.emplace_back(0.0, Eigen::Vector2d{8., 0.});
  poses.emplace_back(0.1, Eigen::Vector2d{1., 7.});
  poses.emplace_back(0.2, Eigen::Vector2d{1., 1.});
  poses.emplace_back(0.3, Eigen::Vector2d{0., 1.});
  poses.emplace_back(0.4, Eigen::Vector2d{0., 0.});
  poses.emplace_back(0.5, Eigen::Vector2d{2., 0.});
  poses.emplace_back(0.6, Eigen::Vector2d{1., 3.});
  poses.emplace_back(0.7, Eigen::Vector2d{4., 1.});
  poses.emplace_back(0.8, Eigen::Vector2d{8., 9.});
  poses.emplace_back(0.9, Eigen::Vector2d{1., 3.});
  poses.emplace_back(0.10, Eigen::Vector2d{1., 6.});
  poses.emplace_back(0.11, Eigen::Vector2d{2., 1.});

  for (auto & p2 : poses) {
    auto p3 = cbr::SE2toSE3(p2);
    auto p22 = cbr::SE3toSE2(p3);

    ASSERT_DOUBLE_EQ(p2.so2().log(), p22.so2().log());
    ASSERT_DOUBLE_EQ(p2.translation()[0], p22.translation()[0]);
    ASSERT_DOUBLE_EQ(p2.translation()[1], p22.translation()[1]);
  }
}
