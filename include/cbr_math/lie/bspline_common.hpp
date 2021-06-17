// Copyright 2020 Yamaha Motor Corporation, USA

#ifndef CBR_MATH__LIE__BSPLINE_COMMON_HPP_
#define CBR_MATH__LIE__BSPLINE_COMMON_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <tuple>

#include "common.hpp"

// forward declare in case we are using ceres
namespace ceres
{
template<typename T, int N>
struct Jet;
}

namespace cbr::lie::bspline
{
using Eigen::Index;

template<class F, class T, class = T>
struct is_static_castable : std::false_type
{
};

template<class F, class T>
struct is_static_castable<F, T, decltype(static_cast<T>(std::declval<F>()))>: std::true_type
{
};

template<typename T>
struct is_ceres_jet : std::false_type
{
};

template<typename Scalar, int N>
struct is_ceres_jet<ceres::Jet<Scalar, N>>: std::true_type
{
};

template<typename S>
double cast_to_double(const S & t)
{
  if constexpr (is_ceres_jet<S>::value) {
    return t.a;
  } else {
    static_assert(
      is_static_castable<S, double>::value,
      "bspline: scalar type must be static_cast-able to a double");
    return static_cast<double>(t);
  }
}

/**
 * Find index idx s.t. breakPts[idx] <= t < breakPts[idx+1] in non-decreasing breakPts
 *
 * Throws an exception if such index does not exist
 */
template<typename S, typename ContainerT>
inline Index findIndex(const S & t_in, const ContainerT & breakPts)
{
  const double t = cast_to_double(t_in);

  if (breakPts.size() < 2 || t < breakPts[0] || t >= breakPts(breakPts.size() - 1)) {
    throw std::runtime_error("BSpline: t not in spline range");
  }

  Index pivot{0}, low{0}, high{breakPts.size() - 1};

  while (low < high) {
    const double frac = (t - breakPts[low]) / (breakPts[high] - breakPts[low]);
    pivot = low + static_cast<Eigen::Index>(frac * (high - low));
    if (t < breakPts[pivot]) {
      high = pivot;
    } else if (t >= breakPts[pivot + 1]) {
      low = pivot + 1;
    } else {
      break;  // we have breakPts[pivot] <= t < breakPts[pivot+1]
    }
  }

  return pivot;
}

/**
 * Calculate beta(idx, j, k) for breakPts.
 * Presence of replicated boundary knot points is presumed.
 */
template<typename ContainerT>
inline double beta(Index idx, Index j, Index k, const ContainerT & breakPts)
{
  // clamp indices when outside interval
  const Index s1 = std::min<Index>(idx - j + k, breakPts.size() - 1);
  const Index s2 = std::max<Index>(idx - j, 0);

  return (breakPts[idx] - breakPts[s2]) / (breakPts[s1] - breakPts[s2]);
}

/**
 * Calculate gamma(idx, j, k) for breakPts.
 * Presence of replicated boundary knot points is presumed.
 */
template<typename ContainerT>
inline double gamma(Index idx, Index j, Index k, const ContainerT & breakPts)
{
  // clamp indices when outside interval
  const Index s1 = std::min<Index>(idx - j + k, breakPts.size() - 1);
  const Index s2 = std::max<Index>(idx - j, 0);

  return (breakPts[idx + 1] - breakPts[idx]) / (breakPts[s1] - breakPts[s2]);
}

/**
 * Calculate B-spline coefficient matrix for breakPts
 *
 * Returns a K+1 x K+1 matrix M s.t.
 * [B_{idx-k}(t)  B_{idx-k+1}(t)  ... B_{idx}(t)] = [1 u ... u^k] M   for u = (t - t_idx) / (t_idx+1 - t_idx).
 *
 * @param idx nominal index to evaluate (generated matrix defines all splines that are nonzero on t_idx, t_{idx+1})
 * @param breakPts vector of knot points, must be non-decreasing
 *
 * Presence of virtual boundary knot points is presumed.
 */
template<Index K, typename ContainerT>
Eigen::Matrix<double, K + 1, K + 1> coeffMatrix(Index idx, const ContainerT & breakPts)
{
  if constexpr (K == 0) {
    static_cast<void>(idx);   // unused
    return Eigen::Matrix<double, 1, 1>::Identity();
  } else {
    // create two matrices with beta and gamma expressions
    Eigen::Matrix<double, K, K + 1> betamat = Eigen::Matrix<double, K, K + 1>::Zero();
    Eigen::Matrix<double, K, K + 1> gammamat = Eigen::Matrix<double, K, K + 1>::Zero();
    for (int i = 0; i != K; ++i) {
      betamat(i, i + 1) = beta(idx, K - i - 1, K, breakPts);
      betamat(i, i) = 1. - betamat(i, i + 1);
      gammamat(i, i) = -gamma(idx, K - i - 1, K, breakPts);
      gammamat(i, i + 1) = -gammamat(i, i);
    }

    Eigen::Matrix<double, K + 1, K + 1> ret = Eigen::Matrix<double, K + 1, K + 1>::Zero();

    // recursive matrix multiplication
    const Eigen::Matrix<double, K, K> coeffs_km1 = coeffMatrix<K - 1>(idx, breakPts);
    ret.template topRows<K>() += coeffs_km1 * betamat;
    ret.template bottomRows<K>() += coeffs_km1 * gammamat;

    return ret;
  }
}

/**
 * Calculate B-spline coefficient matrix for breakPts
 *
 * Returns a K+1 x K+1 matrix M s.t.
 * [B_{idx-k}(t)  B_{idx-k+1}(t)  ... B_{idx}(t)] = [1 u ... u^k] M   for u = (t - t_idx) / (t_idx+1 - t_idx).
 *
 * @param idx nominal index to evaluate (generated matrix defines all splines that are nonzero on t_idx, t_{idx+1})
 * @param breakPts vector of knot points, must be non-decreasing
 *
 * Presence of virtual boundary knot points is presumed.
 */
template<Index K>
Eigen::Matrix<double, K + 1, K + 1> coeffMatrixCard()
{
  if constexpr (K == 0) {
    return Eigen::Matrix<double, 1, 1>::Identity();
  } else {
    // create two matrices with beta and gamma expressions
    Eigen::Matrix<double, K, K + 1> betamat = Eigen::Matrix<double, K, K + 1>::Zero();
    Eigen::Matrix<double, K, K + 1> gammamat = Eigen::Matrix<double, K, K + 1>::Zero();
    for (int i = 0; i != K; ++i) {
      betamat(i, i + 1) = static_cast<double>(K - i - 1) / K;
      betamat(i, i) = 1. - betamat(i, i + 1);
      gammamat(i, i) = -1. / K;
      gammamat(i, i + 1) = -gammamat(i, i);
    }

    Eigen::Matrix<double, K + 1, K + 1> ret = Eigen::Matrix<double, K + 1, K + 1>::Zero();

    // recursive matrix multiplication
    const Eigen::Matrix<double, K, K> coeffs_km1 = coeffMatrixCard<K - 1>();
    ret.template topRows<K>() += coeffs_km1 * betamat;
    ret.template bottomRows<K>() += coeffs_km1 * gammamat;

    return ret;
  }
}

/**
 * Calculate B-spline cumulative coefficient matrix for breakPts
 *
 * Returns a K+1 x K+1 matrix CM s.t.
 * [CB_{idx-k}(t) CB_{idx-k+1}(t) ... CB_{idx}(t)] = [1 u ... u^k] CM   for u = (t - t_idx) / (t_idx+1 - t_idx).
 *
 * The cumulative splines are defined as CB_{i}(t) = \sum_{j=i}^n B_{j}(t).
 *
 * @param idx nominal index to evaluate
 * @param breakPts vector of knot points, must be non-decreasing
 *
 * Presence of virtual boundary knot points is presumed.
 */
template<Index K, typename ContainerT>
Eigen::Matrix<double, K + 1, K + 1> cumCoeffMatrix(int idx, const ContainerT & breakPts)
{
  Eigen::Matrix<double, K + 1, K + 1> ret = coeffMatrix<K>(idx, breakPts);
  for (Index k = K - 1; k >= 0; --k) {
    ret.col(k) += ret.col(k + 1);
  }
  return ret;
}


/**
 * Calculate B-spline cumulative coefficient matrix for evenly space knot points
 *
 * Returns a K+1 x K+1 matrix CM s.t.
 * [CB_{idx-k}(t) CB_{idx-k+1}(t) ... CB_{idx}(t)] = [1 u ... u^k] CM   for u = (t - t_idx) / (t_idx+1 - t_idx).
 *
 * The cumulative splines are defined as CB_{i}(t) = \sum_{j=i}^n B_{j}(t).
 */
template<Index K>
Eigen::Matrix<double, K + 1, K + 1> cumCoeffMatrixCard()
{
  Eigen::Matrix<double, K + 1, K + 1> ret = coeffMatrixCard<K>();
  for (Index k = K - 1; k >= 0; --k) {
    ret.col(k) += ret.col(k + 1);
  }
  return ret;
}


/**
 * @brief Get bspline indices for explicit
 *
 * @param t point on spline to evaluate
 * @param breakPts knot point times
 * @param intrp shift knot points s.t. spline approximately equal to knot points (only correct for odd K)
 *
 * @return (idx, i_beg, i_end, M_i0)
 *
 * idx: index s.t. breakPts[idx] <= t < breakPts[idx+1]
 * i_beg: first index required to evaluate spline
 * i_end: last index to evaluate spline (non-inclusive)
 * M_i0:   column index corresponding to i_beg (typically 1 unless at left end)
 *
 */
template<Index K, typename S, typename ContainerT>
inline std::tuple<Index, Index, Index, Index> getIndices(
  const S & t, const ContainerT & breakPts, bool interp = true
)
{
  const auto idx = bspline::findIndex(t, breakPts);

  auto ibeg0 = idx - K + 1;
  if (interp) {
    ibeg0 += (K - 1) / 2;
  }

  const auto ibeg = std::clamp<Index>(ibeg0, 0, breakPts.size() - 2);
  const auto iend = std::clamp<Index>(ibeg0 + K, 0, breakPts.size() - 1);

  return std::make_tuple(idx, ibeg, iend, 1 + ibeg - ibeg0);
}

/**
 * Calculate the value of a bspline
 *
 *  val0 * \prod_i exp( Bi * diff[i] )
 *
 * where Bi = Mtilde[:, i0 + i] \dot [1; u; u^2; ...; u^{K}]
 *
 * If dv_out and ddv_out are supplied first and second order derivatives
 * are computed. Note that these derivatives are w.r.t. u (which is typically scaled).
 *
 * @param u normalized position on the interval: u = (t - t_idx) / (t_{idx+1} - t_idx)
 * @param Mtilde cumulative coefficient matrix for the interval
 * @param val0 starting value of bspline evaluation
 * @param Mi0 starting  column of Mtilde (typically 1 unless at far left end)
 * @param diff_begin iterator to start of differences
 * @param diff_end iterator pointing to end of differences
 * @param[out] dv_out calculate first order body frame derivative
 * @param[out] ddv_out calculate second order body frame derivative
 */
template<Index K, typename GroupT, typename DiffIterT>
GroupT bspline_eval(
  const typename GroupT::Scalar & u,
  const Eigen::Matrix<typename GroupT::Scalar, K + 1, K + 1> & Mtilde,
  const GroupT & val0,
  Index Mi0,
  DiffIterT diff_begin,
  const DiffIterT & diff_end,
  std::optional<Eigen::Ref<typename GroupT::Tangent>> dv_out = {},
  std::optional<Eigen::Ref<typename GroupT::Tangent>> ddv_out = {}
)
{
  using ScalarT = typename GroupT::Scalar;
  using TngntT = typename GroupT::Tangent;

  GroupT v = val0;
  TngntT dv;
  TngntT ddv;

  if (dv_out || ddv_out) {
    dv.setZero();
  }

  if (ddv_out) {
    ddv.setZero();
  }

  Eigen::Matrix<ScalarT, 1, K + 1> uvec{Eigen::Matrix<ScalarT, 1, K + 1>::UnitX()};
  Eigen::Matrix<ScalarT, 1, K + 1> duvec{Eigen::Matrix<ScalarT, 1, K + 1>::Zero()};
  Eigen::Matrix<ScalarT, 1, K + 1> dduvec{Eigen::Matrix<ScalarT, 1, K + 1>::Zero()};

  for (Index k = 1; k < K + 1; ++k) {
    uvec(k) = uvec(k - 1) * u;
    if (dv_out || ddv_out) {
      duvec(k) = ScalarT(k) * uvec(k - 1);
    }
    if (ddv_out) {
      dduvec(k) = ScalarT(k) * duvec(k - 1);
    }
  }

  for (Index i = Mi0; diff_begin != diff_end; ++diff_begin, ++i) {
    const auto vi = diff_begin->template cast<ScalarT>().eval();

    const ScalarT Bi = uvec.dot(Mtilde.col(i));
    v *= GroupT::exp(Bi * vi);

    if (dv_out || ddv_out) {
      const Eigen::Matrix<ScalarT, GroupT::DoF, GroupT::DoF> Ai = GroupT::exp(-Bi * vi).Adj();
      const ScalarT dBi = duvec.dot(Mtilde.col(i));
      dv.applyOnTheLeft(Ai);
      dv += dBi * vi;

      if (ddv_out) {
        const ScalarT ddBi = dduvec.dot(Mtilde.col(i));
        ddv.applyOnTheLeft(Ai);
        ddv += dBi * GroupT::lieBracket(dv, vi) + ddBi * vi;
      }
    }
  }

  if (dv_out) {
    dv_out = dv;
  }

  if (ddv_out) {
    ddv_out = ddv;
  }

  return v;
}

}  // namespace cbr::lie::bspline

#endif  // CBR_MATH__LIE__BSPLINE_COMMON_HPP_
