// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE


#include <gtest/gtest.h>

#include <sophus/se3.hpp>

#include "cbr_math/lie/common.hpp"
#include "cbr_math/lie/Tn.hpp"
#include "cbr_math/lie/group_product.hpp"


TEST(Lie, Common)
{
  static_assert(
    std::is_same_v<Eigen::Vector2d,
    cbr::lie::detail::change_scalar_eigen<Eigen::Vector2f, double>::type>);
  static_assert(
    std::is_same_v<Eigen::Vector2d,
    cbr::lie::detail::change_scalar_eigen<Eigen::Vector2d, double>::type>);

  static_assert(
    std::is_same_v<Sophus::SE3f,
    cbr::lie::detail::change_scalar_t<Sophus::SE3d, float>>);
  static_assert(
    std::is_same_v<cbr::lie::T5f, cbr::lie::detail::change_scalar_t<cbr::lie::T5d,
    float>>);
  static_assert(
    std::is_same_v<Eigen::Vector4f, cbr::lie::detail::change_scalar_t<Eigen::Vector4d,
    float>>);
}


TEST(Lie, dl_expinv_approx)
{
  using G = Sophus::SO3d;

  Eigen::Vector3d u(1, 2, 3);
  Eigen::Vector3d v(3, 2, 1);

  Eigen::Vector3d dexpinv0 = cbr::lie::dl_expinv_approx<G>(u, v, 0);
  ASSERT_LE((dexpinv0 - v).norm(), 1e-10);

  Eigen::Vector3d dexpinv1 = cbr::lie::dl_expinv_approx<G>(u, v, 1);
  Eigen::Vector3d dexpinv1_ans = v - G::lieBracket(u, v) / 2;
  ASSERT_LE((dexpinv1 - dexpinv1_ans).norm(), 1e-10);

  Eigen::Vector3d dexpinv2 = cbr::lie::dl_expinv_approx<G>(u, v, 2);
  Eigen::Vector3d dexpinv2_ans = v -
    G::lieBracket(u, v) / 2 + G::lieBracket(u, G::lieBracket(u, v)) / 12;
  ASSERT_LE((dexpinv2 - dexpinv2_ans).norm(), 1e-10);

  using G2 = cbr::lie::GroupProduct<double, 0, cbr::lie::T3>;

  Eigen::Vector3d dexpinv3 = cbr::lie::dl_expinv_approx<G2>(u, v, 3);
  ASSERT_LE((dexpinv3 - v).norm(), 1e-10);
}


TEST(Lie, En)
{
  using state_t = cbr::lie::Tn<double, 7>;
  Eigen::Matrix<double, 7, 1> mat;
  mat << 1, 2, 3, 4, 5, 6, 7;
  mat /= 10;

  const state_t state(mat);

  ASSERT_LE((state.translation() - mat).norm(), 1e-10);

  auto cast = state.cast<float>();
  static_assert(
    std::is_same_v<
      std::remove_reference_t<decltype(cast)>,
      cbr::lie::Tn<float, 7>
    >
  );
  ASSERT_LE((cast.translation() - mat.cast<float>()).norm(), 1e-10);

  ASSERT_LE(
    (state.Dx_this_mul_exp_x_at_0() - Eigen::Matrix<double, 7, 7>::Identity()).norm(),
    1e-10);

  ASSERT_LE((state.inverse().translation() + mat).norm(), 1e-10);

  ASSERT_LE((state.log() - mat).norm(), 1e-10);

  ASSERT_LE(((state * state).translation() - 2 * mat).norm(), 1e-10);

  auto copy = state;
  copy *= state;

  ASSERT_LE((copy.translation() - 2 * mat).norm(), 1e-10);

  Eigen::Matrix<double, 7, 1> vec;
  vec << 1, 2, 3, 4, 5, 6, 7;
  vec /= 10;
  ASSERT_LE((state_t::exp(vec).translation() - vec).norm(), 1e-10);

  ASSERT_LE(state_t::lieBracket(vec, vec).norm(), 1e-10);
}


TEST(Lie, EnMap)
{
  using state_t = cbr::lie::T4d;

  double * data = new double[4];
  const double * const_data = data;

  state_t s0{1, 2, 3, 4};

  Eigen::Map<state_t> map(data);
  map = s0;  // copy state->map

  const Eigen::Map<state_t> const_map(data);
  const Eigen::Map<const state_t> const_const_map(const_data);

  ASSERT_DOUBLE_EQ(const_const_map.translation()(0), data[0]);
  ASSERT_DOUBLE_EQ(const_const_map.translation()(1), data[1]);
  ASSERT_DOUBLE_EQ(const_const_map.translation()(2), data[2]);
  ASSERT_DOUBLE_EQ(const_const_map.translation()(3), data[3]);

  state_t copy1(map);  // copy map->state
  copy1 = map;
  ASSERT_DOUBLE_EQ(copy1.translation()(0), data[0]);
  ASSERT_DOUBLE_EQ(copy1.translation()(1), data[1]);
  ASSERT_DOUBLE_EQ(copy1.translation()(2), data[2]);
  ASSERT_DOUBLE_EQ(copy1.translation()(3), data[3]);

  state_t copy2(const_map);  // copy const map<const>->state
  copy2 = const_map;
  ASSERT_DOUBLE_EQ(copy2.translation()(0), data[0]);
  ASSERT_DOUBLE_EQ(copy2.translation()(1), data[1]);
  ASSERT_DOUBLE_EQ(copy2.translation()(2), data[2]);
  ASSERT_DOUBLE_EQ(copy2.translation()(3), data[3]);

  state_t copy3(const_const_map);  // copy const map<const>->state
  copy3 = const_const_map;
  ASSERT_DOUBLE_EQ(copy3.translation()(0), data[0]);
  ASSERT_DOUBLE_EQ(copy3.translation()(1), data[1]);
  ASSERT_DOUBLE_EQ(copy3.translation()(2), data[2]);
  ASSERT_DOUBLE_EQ(copy3.translation()(3), data[3]);

  double * data_copy = new double[4];  // copy map -> map
  Eigen::Map<state_t> map_copy(data_copy);
  map_copy = map;
  ASSERT_DOUBLE_EQ(data_copy[0], data[0]);
  ASSERT_DOUBLE_EQ(data_copy[1], data[1]);
  ASSERT_DOUBLE_EQ(data_copy[2], data[2]);
  ASSERT_DOUBLE_EQ(data_copy[3], data[3]);

  delete[] data_copy;
  delete[] data;
}


TEST(Lie, GroupProduct)
{
  using state_t = cbr::lie::GroupProduct<double, 0, Sophus::SE3, cbr::lie::T4, Sophus::SO3>;

  std::default_random_engine rng(10);

  const Sophus::SE3d part0 = Sophus::SE3d::sampleUniform(rng);
  const Sophus::SO3d part1 = Sophus::SO3d::sampleUniform(rng);

  state_t state_temp;
  std::get<0>(state_temp) = part0;
  std::get<2>(state_temp) = part1;

  state_t state;
  state = state_temp;

  // test assignment
  ASSERT_LE((std::get<0>(state).inverse() * part0).log().norm(), 1e-10);
  ASSERT_LE((std::get<2>(state).inverse() * part1).log().norm(), 1e-10);

  // test adjoint
  auto adjoint = state.Adj();
  ASSERT_LE((adjoint.block<6, 6>(0, 0) - std::get<0>(state).Adj()).norm(), 1e-10);
  ASSERT_LE((adjoint.block<3, 3>(10, 10) - std::get<2>(state).Adj()).norm(), 1e-10);

  // test cast
  auto cast = state.cast<float>();
  static_assert(std::is_same_v<std::remove_reference_t<decltype(std::get<0>(cast))>, Sophus::SE3f>);
  static_assert(std::is_same_v<std::remove_reference_t<decltype(std::get<2>(cast))>, Sophus::SO3f>);
  ASSERT_LE((std::get<0>(cast).inverse() * part0.cast<float>()).log().norm(), 1e-10);
  ASSERT_LE((std::get<2>(cast).inverse() * part1.cast<float>()).log().norm(), 1e-10);

  // test Dx_this_mul_exp_x_at_0
  auto doplus = state.Dx_this_mul_exp_x_at_0();
  ASSERT_LE((doplus.block<7, 6>(0, 0) - std::get<0>(state).Dx_this_mul_exp_x_at_0()).norm(), 1e-10);
  ASSERT_LE(
    (doplus.block<4, 3>(11, 10) - std::get<2>(state).Dx_this_mul_exp_x_at_0()).norm(),
    1e-10);

  // test inverse
  auto inv = state.inverse();
  ASSERT_LE((std::get<0>(inv).inverse() * part0.inverse()).log().norm(), 1e-10);
  ASSERT_LE((std::get<2>(inv).inverse() * part1.inverse()).log().norm(), 1e-10);

  // test log
  auto log = state.log();
  static_assert(decltype(log)::RowsAtCompileTime == 13);
  static_assert(decltype(log)::ColsAtCompileTime == 1);
  ASSERT_LE((log.segment<6>(0) - part0.log()).norm(), 1e-10);
  ASSERT_LE((log.segment<3>(10) - part1.log()).norm(), 1e-10);

  // test mul
  auto mul = state * state;
  ASSERT_LE((std::get<0>(mul).inverse() * (part0 * part0)).log().norm(), 1e-10);
  ASSERT_LE((std::get<2>(mul).inverse() * (part1 * part1)).log().norm(), 1e-10);

  // test in-place mul
  auto copy = state;
  copy *= state;
  ASSERT_LE((std::get<0>(copy).inverse() * (part0 * part0)).log().norm(), 1e-10);
  ASSERT_LE((std::get<2>(copy).inverse() * (part1 * part1)).log().norm(), 1e-10);

  // test exp
  Eigen::Matrix<double, 13,
    1> vec = 0.1 *
    (Eigen::Matrix<double, 13, 1>() << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13).finished();
  auto exp = state_t::exp(vec);
  ASSERT_LE((std::get<0>(exp).inverse() * Sophus::SE3d::exp(vec.head<6>())).log().norm(), 1e-10);
  ASSERT_LE((std::get<2>(exp).inverse() * Sophus::SO3d::exp(vec.tail<3>())).log().norm(), 1e-10);

  // test lieBracket
  Eigen::Matrix<double, 13, 1> vec2 = vec + 0.025 * Eigen::Matrix<double, 13, 1>::Ones();
  auto bracket = state_t::lieBracket(vec, vec2);
  ASSERT_LE(
    (bracket.head<6>() - Sophus::SE3d::lieBracket(
      vec.head<6>(),
      vec2.head<6>())).norm(), 1e-10);
  ASSERT_LE(
    (bracket.tail<3>() - Sophus::SO3d::lieBracket(
      vec.tail<3>(),
      vec2.tail<3>())).norm(), 1e-10);
}


TEST(Lie, GroupProductMapEn)
{
  using state_t = cbr::lie::GroupProduct<double, 0, cbr::lie::T3>;

  double * data1 = new double[state_t::num_parameters];
  double * data2 = new double[state_t::num_parameters];

  Eigen::Map<state_t> map1(data1);
  Eigen::Map<state_t> map2(data2);

  map1 = state_t{Eigen::Vector3d{1, 2, 3}};
  map2 = map1;

  ASSERT_DOUBLE_EQ(data1[0], 1);
  ASSERT_DOUBLE_EQ(data1[1], 2);
  ASSERT_DOUBLE_EQ(data1[2], 3);

  ASSERT_DOUBLE_EQ(data2[0], 1);
  ASSERT_DOUBLE_EQ(data2[1], 2);
  ASSERT_DOUBLE_EQ(data2[2], 3);
}


TEST(Lie, GroupProductMap)
{
  using state_t = cbr::lie::GroupProduct<double, 0, Sophus::SE3, Sophus::SO3, cbr::lie::T3>;

  double * data = new double[state_t::num_parameters];

  Eigen::Map<state_t> map(data);

  std::get<0>(map) = Sophus::SE3d::rotZ(0.5);
  std::get<1>(map) = Sophus::SO3d::rotX(0.5);
  std::get<2>(map) = Eigen::Vector3d{1, 2, 3};

  Eigen::Map<const state_t> const_map(data);
  ASSERT_LE((std::get<0>(const_map).inverse() * Sophus::SE3d::rotZ(0.5)).log().norm(), 1e-10);

  state_t copy1(map);
  ASSERT_LE((std::get<0>(copy1).inverse() * Sophus::SE3d::rotZ(0.5)).log().norm(), 1e-10);
  ASSERT_LE((std::get<1>(copy1).inverse() * Sophus::SO3d::rotX(0.5)).log().norm(), 1e-10);
  ASSERT_LE((std::get<2>(copy1).translation() - Eigen::Vector3d{1, 2, 3}).norm(), 1e-10);

  state_t copy2;
  copy2 = map;
  ASSERT_LE((std::get<0>(copy2).inverse() * Sophus::SE3d::rotZ(0.5)).log().norm(), 1e-10);
  ASSERT_LE((std::get<1>(copy2).inverse() * Sophus::SO3d::rotX(0.5)).log().norm(), 1e-10);
  ASSERT_LE((std::get<2>(copy2).translation() - Eigen::Vector3d{1, 2, 3}).norm(), 1e-10);

  state_t copy3(map);
  ASSERT_LE((std::get<0>(copy3).inverse() * Sophus::SE3d::rotZ(0.5)).log().norm(), 1e-10);
  ASSERT_LE((std::get<1>(copy3).inverse() * Sophus::SO3d::rotX(0.5)).log().norm(), 1e-10);
  ASSERT_LE((std::get<2>(copy3).translation() - Eigen::Vector3d{1, 2, 3}).norm(), 1e-10);

  double * data_copy = new double[state_t::num_parameters];

  // map -> map
  Eigen::Map<state_t> map_copy(data_copy);
  map_copy = map.cast<double>();  // circumvent sophus map -> map copy bug
  ASSERT_LE((std::get<0>(map_copy).inverse() * Sophus::SE3d::rotZ(0.5)).log().norm(), 1e-10);
  ASSERT_LE((std::get<1>(map_copy).inverse() * Sophus::SO3d::rotX(0.5)).log().norm(), 1e-10);
  ASSERT_LE((std::get<2>(map_copy).translation() - Eigen::Vector3d{1, 2, 3}).norm(), 1e-10);

  delete[] data_copy;
  delete[] data;
}


template<typename T>
void run_dr_expinv_test()
{
  static constexpr int DoF = T::DoF;

  std::default_random_engine rng(10);
  std::normal_distribution<double> d;
  auto filler = [&](int) {return d(rng);};

  for (auto i = 0u; i < 10; ++i) {
    Eigen::Matrix<double, DoF, 1> u = 0.05 * Eigen::Matrix<double, DoF, 1>::NullaryExpr(filler);

    auto dr_expinv_u = cbr::lie::dr_expinv<T>(u);
    auto dl_expinv_u = (dr_expinv_u * T::exp(-u).Adj()).eval();

    Eigen::Matrix<double, DoF, DoF> dl_expinv_u_approx;
    for (auto i = 0u; i != DoF; ++i) {
      dl_expinv_u_approx.col(i) =
        cbr::lie::dl_expinv_approx<T>(u, Eigen::Matrix<double, DoF, 1>::Unit(i), 150);
    }

    ASSERT_TRUE(dl_expinv_u.isApprox(dl_expinv_u_approx));

    auto dr_exp_u = cbr::lie::dr_exp<T>(u);
    ASSERT_TRUE(dr_exp_u.isApprox(dr_expinv_u.inverse()));
  }
}


TEST(Lie, dexpinv_closed)
{
  run_dr_expinv_test<cbr::lie::T4d>();
  run_dr_expinv_test<Sophus::SE2d>();
  run_dr_expinv_test<Sophus::SO3d>();
  run_dr_expinv_test<Sophus::SE3d>();
  run_dr_expinv_test<cbr::lie::GroupProduct<double, 0, Sophus::SO3, cbr::lie::T3>>();
}
