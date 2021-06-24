// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE


/*
Cumulative B-splines for interpolation over Lie groups

Changelog
=========
 * 2020-07: Petter Nilsson created first version
 * 2020-09: Moved to lie folder and works with arbitrary groups (Petter)
*
*/

#ifndef CBR_MATH__LIE__BSPLINE_HPP_
#define CBR_MATH__LIE__BSPLINE_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <vector>

#include "bspline_common.hpp"
#include "common.hpp"

namespace cbr::lie
{
namespace bspline_detail
{
using row_t = Eigen::Matrix<double, 1, Eigen::Dynamic>;
}  // namespace bspline_detail

/**
 * Cumulative BSplines on Lie groups
 * @tparam T: Lie group
 * @tparam K: polynomial degree of spline (0: piecewise constant, 1: piecewise linear, 2: quadratic, etc)
 *
 * For knot points `breakPts` and control points `vals` it evaluates
 *
 *  x(t) = vals[0] * \prod_{i=1}^n exp( \tilde B_{i, K}(t) log (vals[i-1].inv() * vals[i]) )
 *
 * where multiplication, log, and exp, are interpreted in the Lie group sense and \tilde B_{i, K} are cumulative
 * B-splines of order K.
 *
 * The type of the evaluation function val() is templated to allow for e.g. automatic differentiation.
 *
 * For interpolation purposes the indices can be shifted via the interp flag so that
 * $B_{i, K}$ is matched with vals[i-1+(K+1)/2].inv() * vals[i+(K+1)/2] since that is
 * the index where $B_{i, K}(t)$ achieves its maximum.
 */
template<typename T, Eigen::Index K = 3>
class BSpline
{
  using Scalar = typename T::Scalar;
  using Tangent = typename T::Tangent;
  using DiffListT = std::vector<Tangent, Eigen::aligned_allocator<Tangent>>;
  using CoeffMatT = Eigen::Matrix<double, K + 1, K + 1>;

  using Index = Eigen::Index;

public:
  using row_t = bspline_detail::row_t;
  using ValListT = std::vector<T, Eigen::aligned_allocator<T>>;
  /**
   * @brief Default-construct a new BSpline object that defines the constant
   * mapping t -> Identity on t \in [0, 1]
   */
  BSpline()
  : breakPts_((row_t(2) << 0, 1).finished()), vals_(ValListT{T{}, T{}}), card_(false), interp_(true)
  {
  }

  /**
   * Create BSpline for given knots t_i and vals x_i s.t. x(t_i) \approx x_i
   * @param breakPts knot points
   * @param vals control points
   * @param card use optimized cardinal bspline, assumes that knots are equally spaced
   * @param interp shift vals so that control points coincide with spline maximum.
   *               important to avoid time shift when using the spline for interpolation,
   *               but only works with odd-degree splines
   */
  BSpline(row_t && breakPts, ValListT && vals, bool card = false, bool interp = true)
  : breakPts_(breakPts), vals_(vals), card_(card), interp_(interp)
  {
    const Index n = breakPts_.size();

    if (interp_ && K % 2 == 0) {
      throw std::runtime_error("interp=true requires odd spline degree");
    }

    if (n < 2u || n != static_cast<Index>(vals.size())) {
      throw std::runtime_error("Required: breakPts.size() == vals.size() >= 2");
    }

    // calculate difference vector
    diffs_.reserve(n - 1);
    for (auto i = 0u; i < n - 1; ++i) {
      diffs_.push_back((vals_[i].inverse() * vals_[i + 1]).log());
    }

    // compute cumulative coefficient matrices
    if (!card_ || n - 2 <= 2 * K - 3) {
      // not equally spaced or not enough points to expose constant middle
      coef_mats_.reserve(n - 1);
      for (auto idx = 0u; idx < n - 1; ++idx) {
        coef_mats_.push_back(bspline::cumCoeffMatrix<K>(idx, breakPts_));
      }
    } else {
      // Index helper   <-- head --> <- middle ->  <---   tail   --->  (invalid)
      // breakPts_:   [ 0 1  ... K-2 K-1 ...  N-K  N-K+1  ...  N-2      N-1     ] size n
      // coef_mats_:  [ 0 1  ... K-2 K-1 ...  K-1  K      ...  2*K-3            ] size 2*(K-1)

      coef_mats_.reserve(2 * K - 2);
      for (auto idx = 0u; idx < K - 1; ++idx) {  // k-1 head matrices
        coef_mats_.push_back(bspline::cumCoeffMatrix<K>(idx, breakPts_));
      }
      // middle (constant) matrix
      coef_mats_.push_back(bspline::cumCoeffMatrixCard<K>());
      for (auto idx = n - K + 1; idx < n - 1; ++idx) {  // k-2 tail matrices
        coef_mats_.push_back(bspline::cumCoeffMatrix<K>(idx, breakPts_));
      }
    }
  }

  /**
   * Create BSpline for given knots t_i and vals x_i s.t. x(t_i) \approx x_i
   * @param breakPts knot points
   * @param vals control points in container that supports begin() and end()
   * @param card use optimized cardinal bspline, assumes that knots are equally spaced
   * @param interp shift vals so that control points coincide with spline maximum.
   *               important to avoid time shift when using the spline for interpolation,
   *               but only works with odd-degree splines
   */
  template<typename Container>
  BSpline(const row_t & breakPts, const Container & vals, bool card = false, bool interp = true)
  : BSpline(row_t(breakPts), ValListT(vals.cbegin(), vals.cend()), card, interp)
  {}

  BSpline(const BSpline &) = default;
  BSpline(BSpline &&) = default;
  BSpline & operator=(const BSpline &) = default;
  BSpline & operator=(BSpline &&) = default;

  /**
   * Check if spline is cardinal
   */
  inline bool is_cardinal() const {return card_;}

  /**
   * Spline starting time
   */
  inline double min_time() const {return breakPts_.head<1>()(0);}

  /**
   * Spline ending time
   */
  inline double max_time() const {return breakPts_.tail<1>()(0);}

  /**
   * Check if t is within bounds of spline
   */
  template<typename S = typename T::Scalar>
  inline bool covers(S t_in) const
  {
    const double t = bspline::cast_to_double(t_in);

    return breakPts_.size() > 0 && min_time() <= static_cast<double>(t) &&
           static_cast<double>(t) < max_time();
  }

  /**
   * Get time range of data points that define the spline value at a given point t
   */
  template<typename S = typename T::Scalar>
  inline double support(S t) const
  {
    return support_idx(bspline::findIndex(t, breakPts_));
  }

  /**
   * Returns true if all points within an interval are covered by the spline and no point within the
   * interval has a support larger than max_support
   */
  template<typename S = typename T::Scalar>
  inline bool interval_quality(S t0, S t1, double max_support) const
  {
    if (!covers(t0) || !covers(t1)) {
      return false;
    }
    const auto idx0 = bspline::findIndex(t0, breakPts_);
    const auto idx1 = bspline::findIndex(t1, breakPts_);
    for (auto idx = idx0; idx <= idx1; ++idx) {
      if (support_idx(idx) > max_support) {
        return false;
      }
    }
    return true;
  }

  /**
   * Evaluate spline at a point
   * @tparam S type to use (e.g. double or autodiff type)
   * @param t parameter value to evaluate
   */
  template<typename S = typename T::Scalar>
  typename detail::change_scalar_t<T, S> val(const S & t) const
  {
    const auto[idx, i_beg, i_end, Mi0] = bspline::getIndices<K>(t, breakPts_, interp_);
    const auto Mtilde = retrieveCoefMat_(idx).template cast<S>();
    const S delta_t = S(breakPts_(idx + 1) - breakPts_(idx));
    const S u = (t - breakPts_(idx)) / delta_t;

    return bspline::bspline_eval<K>(
      u, Mtilde, vals_[i_beg].template cast<S>(), Mi0,
      diffs_.begin() + i_beg, diffs_.begin() + i_end);
  }

  /**
   * Evaluate body tangent space velocity at a point
   *
   * Returns \omega \in TM s.t. \dot X = X \hat \omega where X is
   * the matrix representation of the Lie group element X
   *
   * @tparam S type to use (e.g. double or autodiff type)
   */
  template<typename S = typename T::Scalar>
  typename detail::change_scalar_t<typename T::Tangent, S> der(const S & t) const
  {
    using TngntT = typename detail::change_scalar_t<typename T::Tangent, S>;

    const auto[idx, i_beg, i_end, Mi0] = bspline::getIndices<K>(t, breakPts_, interp_);
    const auto Mtilde = retrieveCoefMat_(idx).template cast<S>();
    const S delta_t = S(breakPts_(idx + 1) - breakPts_(idx));
    const S u = (t - breakPts_(idx)) / delta_t;

    TngntT dv;
    bspline::bspline_eval<K>(
      u, Mtilde, vals_[i_beg].template cast<S>(), Mi0,
      diffs_.begin() + i_beg, diffs_.begin() + i_end, dv);
    return dv / delta_t;
  }

  /**
   * Evaluate body tangent space acceleration at a point
   *
   * Returns \omega \in TM s.t. \ddot X = X (\hat \omega)^2 + X \hat \dot \omega
   * where X is the matrix representation of the Lie group element X
   *
   * @tparam S type to use (e.g. double or autodiff type)
   */
  template<typename S = typename T::Scalar>
  typename detail::change_scalar_t<typename T::Tangent, S> der2(const S & t) const
  {
    using TngntT = typename detail::change_scalar_t<typename T::Tangent, S>;

    const auto[idx, i_beg, i_end, Mi0] = bspline::getIndices<K>(t, breakPts_, interp_);
    const auto Mtilde = retrieveCoefMat_(idx).template cast<S>();
    const S delta_t = S(breakPts_(idx + 1) - breakPts_(idx));
    const S u = (t - breakPts_(idx)) / delta_t;

    TngntT d2v;
    bspline::bspline_eval<K>(
      u, Mtilde, vals_[i_beg].template cast<S>(), Mi0,
      diffs_.begin() + i_beg, diffs_.begin() + i_end, {}, d2v);
    return d2v / delta_t / delta_t;
  }

protected:
  /**
   * Retrieve pre-computed cumulative coefficient matrix for idx
   */
  inline const CoeffMatT & retrieveCoefMat_(Index idx) const
  {
    // Index helper   <-- head --> <- middle ->  <---   tail   --->  (invalid)
    // breakPts_:   [ 0 1  ... K-2 K-1 ...  N-K  N-K+1  ...  N-2      N-1     ] size n
    // coef_mats_:  [ 0 1  ... K-2 K-1 ...  K-1  K      ...  2*K-3            ] size 2*(K-1)
    Index n = breakPts_.size();

    if (!card_ || n - 2 <= 2 * K - 3) {
      return coef_mats_[idx];
    } else if (idx < K - 1) {
      return coef_mats_[idx];  // head
    } else if (idx > n - K) {
      return coef_mats_[idx - (n + 1 - 2 * K)];  // tail
    } else {
      return coef_mats_[K - 1];  // middle
    }
  }

  /**
   * If interp_, shift index in breakPts s.t. B_i(t) achieves its maximum at new index
   */
  inline Index val_idx(Index i) const
  {
    if (interp_) {
      return i + (K + 1) / 2;
    }
    return i;
  }

  /**
   * Return the support for splines inside an interval
   */
  inline double support_idx(Index idx) const
  {
    return breakPts_(std::clamp<Index>(val_idx(idx), 0, breakPts_.size() - 1)) -
           breakPts_(std::clamp<Index>(val_idx(idx - K), 0, breakPts_.size() - 1));
  }

  // time breakpoints, does not contain repeated endpoints
  row_t breakPts_;

  // vals, same length as breakPts_
  ValListT vals_;

  // sequential differences of vals_
  DiffListT diffs_;

  // pre-computed cumulative coefficient matrices
  // contains n - 1 elements if card = false (one for each index)
  // contains 2 * (k-1) + 1 elements if card = true since inner matrices are constant
  std::vector<CoeffMatT, Eigen::aligned_allocator<CoeffMatT>> coef_mats_;

  // true if breakPts_ is cardinal, i.e. equally spaced
  bool card_;

  // if true, shift control points to align with spline maxima when evaluating the spline
  bool interp_;
};

// template deduction guides
template<typename Container>
BSpline(typename bspline_detail::row_t &, Container &)
->BSpline<typename Container::value_type>;

template<typename Container>
BSpline(typename bspline_detail::row_t &, Container &, bool)
->BSpline<typename Container::value_type>;

template<typename Container>
BSpline(typename bspline_detail::row_t &, Container &, bool, bool)
->BSpline<typename Container::value_type>;

}  // namespace cbr::lie


#endif  // CBR_MATH__LIE__BSPLINE_HPP_
