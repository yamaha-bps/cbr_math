// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_MATH__INTERP__PIECEWISE_LINEAR_HPP_
#define CBR_MATH__INTERP__PIECEWISE_LINEAR_HPP_

#include <Eigen/Dense>

#include <utility>
#include <vector>
#include <array>

#include "piecewise_poly.hpp"

namespace cbr
{

class PiecewiseLinear
{
protected:
  using row_t = Eigen::Matrix<double, 1, Eigen::Dynamic>;
  using matrix_t = Eigen::MatrixXd;

public:
  /**
   * @brief Create a scalar-valued piecewise linear polynomial
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 PiecewisePoly::row_t
   * @param x sorted breakpoints
   * @param y values
   *
   * The resulting function f is s.t.
   *  f(t) = y[i] + alpha * (y[i+1] - y[i]) if x[i] <= t < x[i+1]
   *     where alpha = (t - x[i]) / (x[i+1] - x[i])
   *  f(t) = y[0] if t < x[0]
   *  f(t) = y[x.size() - 1] if t >= x[x.size() - 1]
   */
  template<typename T1, typename T2>
  static PiecewisePoly fit(
    T1 && x,
    const Eigen::DenseBase<T2> & y)
  {
    static_assert(is_eigen_dense_v<T1>, "x must be an Eigen::DenseBase object.");

    auto coeffs = generateCoeffs(x, y);
    return PiecewisePoly(std::forward<T1>(x), std::move(coeffs));
  }

  /**
   * @brief Create a vector-valued piecewise linear polynomial
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 PiecewisePoly::matrix_t or container_t<PiecewisePoly::row_t>
   * @param x sorted breakpoints
   * @param y values
   *
   * The resulting function f is s.t.
   *  f(t) = y[i] + alpha * (y[i+1] - y[i]) if x[i] <= t < x[i+1]
   *     where alpha = (t - x[i]) / (x[i+1] - x[i])
   *  f(t) = y[0] if t < x[0]
   *  f(t) = y[x.size() - 1] if t >= x[x.size() - 1]
   */
  template<typename T1, typename T2>
  static PiecewisePolyND fitND(
    T1 && x,
    const T2 & ys)
  {
    static_assert(is_eigen_dense_v<T1>, "x must be an Eigen::DenseBase object.");

    if (x.size() < 2) {
      throw std::invalid_argument("x must be of size > 1.");
    }

    std::vector<matrix_t> coefLists;

    if constexpr (is_eigen_dense_v<T2>) {
      if (ys.rows() < 1) {
        throw std::invalid_argument("Dimension of the data must be > 0.");
      }

      if (ys.cols() != x.size()) {
        throw std::invalid_argument("The number of columns of ys must be equal to the size of x.");
      }
      coefLists.reserve(static_cast<std::size_t>(ys.rows()));

      for (Eigen::Index i = 0; i < ys.rows(); i++) {
        coefLists.push_back(generateCoeffs(x, ys.row(i)));
      }
    } else {
      static_assert(
        is_eigen_dense_v<typename T2::value_type>&& T2::value_type::IsVectorAtCompileTime,
        "ys must be a container of Eigen::DenseBase vector objects");

      if (ys.size() < 1) {
        throw std::invalid_argument("Dimension of the data must be > 0.");
      }

      coefLists.reserve(ys.size());

      for (const auto & y : ys) {
        coefLists.push_back(generateCoeffs(x, y));
      }
    }

    return PiecewisePolyND(std::forward<T1>(x), std::move(coefLists));
  }

protected:
  template<typename T1, typename T2>
  static matrix_t generateCoeffs(
    const Eigen::DenseBase<T1> & x,
    const Eigen::DenseBase<T2> & y)
  {
    static_assert(
      T1::IsVectorAtCompileTime && T2::IsVectorAtCompileTime,
      "x and y must be vectors.");

    if (x.size() != y.size()) {
      throw std::invalid_argument("Each element of ys must have the same size as x.");
    }

    Eigen::Index nj = x.size() - 1;
    matrix_t coeffs(2, nj);

    for (Eigen::Index i = 0; i < nj; i++) {
      coeffs(1, i) = y[i];
      coeffs(0, i) = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }

    return coeffs;
  }
};

}  // namespace cbr

#endif  // CBR_MATH__INTERP__PIECEWISE_LINEAR_HPP_
