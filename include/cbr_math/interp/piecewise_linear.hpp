// Copyright 2020 Yamaha Motor Corporation, USA
#ifndef CBR_MATH__INTERP__PIECEWISE_LINEAR_HPP_
#define CBR_MATH__INTERP__PIECEWISE_LINEAR_HPP_

#include <Eigen/Dense>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

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
      coefLists.reserve(ys.rows());

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

  /**
   * @brief Create a Lie-valued piecewise linear polynomial
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 container_t<LieType>
   * @param x sorted breakpoints
   * @param y values
   */
  template<typename T1, typename T2>
  static auto fitLie(
    T1 && x,
    T2 && poses)
  {
    static_assert(is_eigen_dense_v<T1>, "x must be an Eigen::DenseBase object.");

    using Group = typename std::decay_t<T2>::value_type;
    static_assert(
      std::is_base_of_v<Sophus::SO2Base<Group>, Group>||
      std::is_base_of_v<Sophus::SE2Base<Group>, Group>||
      std::is_base_of_v<Sophus::SO3Base<Group>, Group>||
      std::is_base_of_v<Sophus::SE3Base<Group>, Group>,
      "Group must be one of Sophus' SO2, SO3, SE2 or SE3.");

    using GroupScalar = typename Group::Scalar;
    constexpr static int Dimension = Group::DoF;

    static_assert(std::is_same_v<GroupScalar, double>, "Group scalar must be a double.");

    if (x.size() < 2) {
      throw std::invalid_argument("x be of size > 1");
    }

    if (static_cast<std::size_t>(x.size()) != poses.size()) {
      throw std::invalid_argument("x and poses must have the same size.");
    }

    Eigen::Matrix<double, Dimension, Eigen::Dynamic> omegas(Dimension, (poses.size() - 1));
    for (std::size_t i = 0; i < (poses.size() - 1); i++) {
      omegas.col(i) =
        Eigen::Matrix<double, Dimension, 1>((poses[i].inverse() * poses[i + 1]).log());
    }

    std::vector<matrix_t> coefLists;
    coefLists.reserve(Dimension);
    for (int i = 0; i < Dimension; i++) {
      coefLists.push_back(generateCoeffsLie(x, omegas.row(i)));
    }

    return PiecewisePolyLie<Group>(
      std::forward<T1>(x),
      std::move(coefLists),
      std::forward<T2>(poses));
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

    std::size_t nj = x.size() - 1;
    matrix_t coeffs(2, nj);

    for (std::size_t i = 0; i < nj; i++) {
      coeffs(1, i) = y[i];
      coeffs(0, i) = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }

    return coeffs;
  }

  template<typename T1, typename T2>
  static matrix_t generateCoeffsLie(
    const Eigen::DenseBase<T1> & x,
    const Eigen::DenseBase<T2> & omega)
  {
    static_assert(
      T1::IsVectorAtCompileTime && T2::IsVectorAtCompileTime,
      "x and omega must be vectors.");

    if (x.size() != omega.size() + 1) {
      throw std::invalid_argument("Size of x must be equal to size of omega + 1.");
    }

    std::size_t nj = x.size() - 1;
    matrix_t coeffs(2, nj);

    for (std::size_t i = 0; i < nj; i++) {
      coeffs(1, i) = 0.;
      coeffs(0, i) = omega[i] / (x[i + 1] - x[i]);
    }

    return coeffs;
  }
};

}  // namespace cbr

#endif  // CBR_MATH__INTERP__PIECEWISE_LINEAR_HPP_
