// Copyright 2020 Yamaha Motor Corporation, USA
#ifndef CBR_MATH__INTERP__PIECEWISE_CONSTANT_HPP_
#define CBR_MATH__INTERP__PIECEWISE_CONSTANT_HPP_

#include <Eigen/Dense>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <utility>
#include <vector>
#include <array>

#include "piecewise_poly.hpp"

namespace cbr
{

class PiecewiseConstant
{
protected:
  using row_t = Eigen::Matrix<double, 1, Eigen::Dynamic>;
  using matrix_t = Eigen::MatrixXd;

public:
  /**
   * @brief Create a scalar-valued piecewise constant polynomial
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 PiecewisePoly::row_t
   * @param x sorted breakpoints
   * @param y values
   *
   * The resulting function f is s.t.
   *  f(t) = y[i] if x[i] <= t < x[i+1]
   *  f(t) = y[0] if t < x[0]
   *  f(t) = y[x.size() - 1] if t >= x[x.size() - 1]
   */
  template<typename T1, typename T2>
  static PiecewisePoly fit(
    T1 && x,
    const Eigen::DenseBase<T2> & y)
  {
    static_assert(is_eigen_dense_v<T1>, "x must be an Eigen::DenseBase object.");

    // add a dummy datapoint at the end
    const auto newsize = x.size() + 1;
    row_t x_ext(newsize);
    x_ext << x, x(x.size() - 1) + 1;
    row_t y_ext(newsize);
    y_ext << y, y(y.cols() - 1);

    auto coeffs = generateCoeffs(x_ext, y_ext);
    return PiecewisePoly(std::move(x_ext), std::move(coeffs));
  }

  /**
   * @brief Create a vector-valued piecewise constant polynomial
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 PiecewisePoly::matrix_t or container_t<PiecewisePoly::row_t>
   * @param x sorted breakpoints
   * @param y values
   *
   * The resulting function f is s.t.
   *  f(t) = y[i] if x[i] <= t < x[i+1]
   *  f(t) = y[0] if t < x[0]
   *  f(t) = y[x.size() - 1] if t >= x[x.size() - 1]
   */
  template<typename T1, typename T2>
  static PiecewisePolyND fitND(
    T1 && x,
    const T2 & ys)
  {
    static_assert(is_eigen_dense_v<T1>, "x must be an Eigen::DenseBase object.");

    if (x.size() == 0) {
      throw std::invalid_argument("x must be non-empty");
    }

    // add a dummy datapoint at the end
    const auto newsize = x.size() + 1;
    row_t x_ext(newsize);
    x_ext << x, x(x.size() - 1) + 1;

    std::vector<matrix_t> coefLists;

    if constexpr (is_eigen_dense_v<T2>) {
      if (ys.rows() < 1) {
        throw std::invalid_argument("Dimension of the data must be > 0.");
      }

      if (ys.cols() != x.size()) {
        throw std::invalid_argument("The number of columns of ys must be equal to the size of x.");
      }

      // add a dummy datapoint at the end
      matrix_t ys_ext(ys.rows(), newsize);
      ys_ext << ys, ys.col(ys.cols() - 1);

      coefLists.reserve(ys_ext.rows());

      for (Eigen::Index i = 0; i < ys_ext.rows(); i++) {
        coefLists.push_back(generateCoeffs(x_ext, ys_ext.row(i)));
      }
    } else {
      static_assert(
        is_eigen_dense_v<typename T2::value_type>&& T2::value_type::IsVectorAtCompileTime,
        "ys must be a container of Eigen::DenseBase vector objects");

      if (ys.size() < 1) {
        throw std::invalid_argument("Dimension of the data must be > 0.");
      }

      coefLists.reserve(ys.size());

      for (auto & y : ys) {
        // add a dummy datapoint at the end
        row_t y_ext(newsize);
        y_ext << y, y(y.size() - 1);
        coefLists.push_back(generateCoeffs(x_ext, std::move(y_ext)));
      }
    }

    return PiecewisePolyND(std::move(x_ext), std::move(coefLists));
  }

  /**
   * @brief Create a Lie-valued piecewise constant polynomial
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 container_t<LieType>
   * @param x sorted breakpoints
   * @param y values
   *
   * NOTE: this function is not tested
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

    if (x.size() == 0) {
      throw std::invalid_argument("x must be non-empty");
    }

    if (static_cast<std::size_t>(x.size()) != poses.size()) {
      throw std::invalid_argument("x and poses must have the same size.");
    }

    // add a dummy datapoint at the end
    const auto newsize = x.size() + 1;
    row_t x_ext(newsize);
    x_ext << x, x(x.size() - 1) + 1;

    std::vector<Group> poses_ext(newsize);
    for (auto i = 0u; i != poses.size(); ++i) {
      poses_ext[i] = poses[i];
    }
    poses_ext.back() = poses.back();

    std::vector<matrix_t> coefLists;
    coefLists.reserve(Dimension);
    for (int i = 0; i < Dimension; i++) {
      coefLists.push_back(generateCoeffsLie(x_ext));
    }

    return PiecewisePolyLie<Group>(
      std::move(x_ext),
      std::move(coefLists),
      std::move(poses_ext));
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
    matrix_t coeffs(1, nj);

    for (std::size_t i = 0; i < nj; i++) {
      coeffs(0, i) = y[i];
    }

    return coeffs;
  }

  template<typename T1>
  static matrix_t generateCoeffsLie(
    const Eigen::DenseBase<T1> & x)
  {
    static_assert(
      T1::IsVectorAtCompileTime,
      "x and omega must be vectors.");

    std::size_t nj = x.size() - 1;
    matrix_t coeffs(1, nj);
    coeffs.setZero();

    return coeffs;
  }
};

}  // namespace cbr

#endif  // CBR_MATH__INTERP__PIECEWISE_CONSTANT_HPP_
