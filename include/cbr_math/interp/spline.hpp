// Copyright 2020 Yamaha Motor Corporation, USA
#ifndef CBR_MATH__INTERP__SPLINE_HPP_
#define CBR_MATH__INTERP__SPLINE_HPP_

#include <Eigen/Dense>

#include <utility>
#include <vector>
#include <array>

#include "piecewise_poly.hpp"

namespace cbr
{

class Spline
{
protected:
  using row_t = Eigen::Matrix<double, 1, Eigen::Dynamic>;
  using matrix_t = Eigen::MatrixXd;

public:
  /**
   * @brief Create a scalar-valued spline
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 PiecewisePoly::matrix_t
   * @param x breakpoints of shape 1xN
   * @param y values of shape DxN where D \in {1, 2, 3}
   *
   * D=1: y contains values
   * D=2: y contains values, 1st derivatives
   * D=3: y contains values, 1st derivatives, 2nd derivatives
   */
  template<typename T1, typename T2>
  static PiecewisePoly fit(
    T1 && x,
    const Eigen::DenseBase<T2> & y)
  {
    static_assert(is_eigen_dense_v<T1>, "x must be an Eigen::DenseBase object.");

    if (x.size() < 2) {
      throw std::invalid_argument("x be of size > 1");
    }

    if (x.size() != y.cols()) {
      throw std::invalid_argument("The number of columns of ys must be equal to the size of x.");
    }

    if (!(y.rows() > 0 && y.rows() < 4)) {
      throw std::invalid_argument("y must have between 1 and 3 rows.");
    }

    matrix_t coeffs = generateCoeffs(x, y);

    if (y.rows() == 1 && x.size() == 3) {
      Eigen::Matrix<double, 1, 2> bp{x[0], x[2]};
      return PiecewisePoly(std::move(bp), std::move(coeffs));
    } else {
      return PiecewisePoly(std::forward<T1>(x), std::move(coeffs));
    }
  }

  /**
   * @brief Create a vector-valued spline
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 container_t<PiecewisePoly::matrix_t>
   * @param x breakpoints of shape 1xN
   * @param y values of shape MxDxN where M is number of value dimensions and D \in {1, 2, 3}
   *
   * D=1: y[m] contains values for m:th dimension
   * D=2: y[m] contains values, 1st derivatives for m:th dimension
   * D=3: y[m] contains values, 1st, 2nd derivatives for m:th dimension
   */
  template<typename T1, typename T2>
  static PiecewisePolyND fitND(
    T1 && x,
    const T2 & ys)
  {
    static_assert(is_eigen_dense_v<T1>, "x must be an Eigen::DenseBase object.");

    if (x.size() < 2) {
      throw std::invalid_argument("x must be of size > 1");
    }

    if constexpr (is_eigen_dense_v<T2>) {
      if (ys.rows() < 1) {
        throw std::invalid_argument("Number of rows of ys must be > 0.");
      }

      if (x.size() != ys.cols()) {
        throw std::invalid_argument("The number of columns of ys must be equal to the size of x.");
      }

      std::vector<matrix_t> coefLists;
      coefLists.reserve(ys.rows());

      for (Eigen::Index i = 0; i < ys.rows(); i++) {
        coefLists.push_back(generateCoeffs(x, ys.row(i)));
      }

      return PiecewisePolyND(std::forward<T1>(x), std::move(coefLists));
    } else {
      static_assert(
        is_eigen_dense_v<typename T2::value_type>,
        "ys must be a container of Eigen::DenseBase objects");

      if (ys.size() < 1) {
        throw std::invalid_argument("Number of elements of ys must be > 0.");
      }

      const auto ni = ys[0].rows();
      if (!(ni > 0 && ni < 4)) {
        throw std::invalid_argument("Each element of ys must have between 1 and 3 rows.");
      }
      for (const auto & y : ys) {
        if (y.rows() != ni) {
          throw std::invalid_argument(
                  "Each element of ys must have the same number of rows.");
        }
      }

      std::vector<matrix_t> coefLists;
      coefLists.reserve(ys.size());
      for (const auto & y : ys) {
        coefLists.push_back(generateCoeffs(x, y));
      }

      if (ni == 1 && x.size() == 3) {
        Eigen::Matrix<double, 1, 2> bp{x[0], x[2]};
        return PiecewisePolyND(std::move(bp), std::move(coefLists));
      } else {
        return PiecewisePolyND(std::forward<T1>(x), std::move(coefLists));
      }
    }
  }

  /**
   * @brief Create a Lie-valued spline
   *
   * @tparam T1 Eigen::VectorXd
   * @tparam T2 container_t<LieType> where LieType \in {SO2, SO3, SE2, SE3}
   * @param x breakpoints of shape 1xN
   * @param y values of shape 1xN
   */
  template<typename T1, typename T2>
  static auto fitLie(
    T1 && x,
    T2 && poses)
  {
    static_assert(is_eigen_dense_v<T1>, "x must be an Eigen::DenseBase object.");

    using Group = typename std::decay_t<T2>::value_type;

    using Tangent = typename Group::Tangent;
    using GroupScalar = typename Group::Scalar;
    constexpr static int Dimension = Group::DoF;

    static_assert(std::is_same_v<GroupScalar, double>, "Group scalar must be a double.");

    const auto N = x.size();

    if (N < 2) {
      throw std::invalid_argument("x must be of size > 1");
    }

    if (static_cast<std::size_t>(x.size()) != poses.size()) {
      throw std::invalid_argument("x and poses must have the same size.");
    }

    Eigen::Matrix<double, Dimension, std::decay_t<T1>::SizeAtCompileTime> vels(Dimension, N);

    Tangent logNm1 = (poses[0].inverse() * poses[1]).log();
    vels.col(0) = Eigen::Matrix<double, Dimension, 1>(logNm1);
    vels.col(N - 1) =
      Eigen::Matrix<double, Dimension, 1>((poses[N - 2].inverse() * poses[N - 1]).log());

    if (N > 2) {
      for (std::decay_t<decltype(N)> i = 1; i < N - 2; i++) {
        const Tangent logN = (poses[i].inverse() * poses[i + 1]).log();
        vels.col(i) = Eigen::Matrix<double, Dimension, 1>(0.5 * (logN + logNm1));
        logNm1 = logN;
      }
    }

    return fitLie(std::forward<T1>(x), std::forward<T2>(poses), std::move(vels));
  }


  /**
   * @brief Create a Lie-valued spline with given derivatives
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 container_t<LieType>
   * @tparam T3 container_t<LieType::Tangent> or PiecewisePoly::matrix_t
   * @param x breakpoints of shape 1xN
   * @param poses values of shape 1xN
   * @param vels velocities of shape DoFxN
   */
  template<typename T1, typename T2, typename T3>
  static auto fitLie(
    T1 && x,
    T2 && poses,
    const T3 & vels)
  {
    static_assert(is_eigen_dense_v<T1>, "x must be an Eigen::DenseBase object.");

    using Group = typename std::decay_t<T2>::value_type;

    using Tangent = typename Group::Tangent;
    using GroupScalar = typename Group::Scalar;
    constexpr static int Dimension = Group::DoF;

    static_assert(std::is_same_v<GroupScalar, double>, "Group scalar must be a double.");

    if (x.size() < 2) {
      throw std::invalid_argument("x must be of size > 1");
    }

    if (static_cast<std::size_t>(x.size()) != poses.size()) {
      throw std::invalid_argument("x and poses must have the same size.");
    }

    std::vector<matrix_t> coefLists;
    coefLists.reserve(Dimension);

    Eigen::Matrix<double, Dimension, Eigen::Dynamic> omegas(Dimension, (poses.size() - 1));
    Eigen::Matrix<double, Dimension, Eigen::Dynamic> velNp1(Dimension, (poses.size() - 1));

    if constexpr (is_eigen_dense_v<T3>) {
      static_assert(
        T3::RowsAtCompileTime == Dimension, "vels must have as many rows as the "
        "dimension of the tangent space of the lie group of poses.");

      if (x.size() != vels.cols()) {
        throw std::invalid_argument(
                "The number of columns of vels must be equal to the size of x.");
      }

      for (std::size_t i = 0; i < (poses.size() - 1); i++) {
        const auto dPose = poses[i].inverse() * poses[i + 1];
        omegas.col(i) = Eigen::Matrix<double, Dimension, 1>(dPose.log());

        if constexpr (Dimension == 1) {
          velNp1[i] = vels[i + 1];
        } else {
          const auto dExp = lie::dr_exp<Group>(omegas.col(i));
          velNp1.col(i) = dExp.fullPivLu().solve(vels.col(i + 1));
        }
      }

      for (Eigen::Index i = 0; i < Dimension; i++) {
        coefLists.push_back(generateCoeffsLie(x, omegas.row(i), vels.row(i), velNp1.row(i)));
      }
    } else {
      static_assert(
        std::is_same_v<typename T3::value_type, Tangent>,
        "vels must be a container of elements of the tangent space of the lie group of poses.");

      if (static_cast<std::size_t>(x.size()) != vels.size()) {
        throw std::invalid_argument("x and vels must have the same size.");
      }

      for (std::size_t i = 0; i < (poses.size() - 1); i++) {
        const auto dPose = poses[i].inverse() * poses[i + 1];
        omegas.col(i) = Eigen::Matrix<double, Dimension, 1>(dPose.log());

        if constexpr (Dimension == 1) {
          velNp1[i] = vels[i + 1];
        } else {
          const auto dExp = lie::dr_exp<Group>(omegas.col(i));
          velNp1.col(i) = dExp.fullPivLu().solve(vels[i + 1]);
        }
      }

      Eigen::Matrix<double, Dimension, Eigen::Dynamic> velsMatrix(Dimension, vels.size());
      for (std::size_t i = 0; i < vels.size(); i++) {
        velsMatrix.col(i) = Eigen::Matrix<double, Dimension, 1>(vels[i]);
      }


      for (Eigen::Index i = 0; i < Dimension; i++) {
        coefLists.push_back(generateCoeffsLie(x, omegas.row(i), velsMatrix.row(i), velNp1.row(i)));
      }
    }

    return PiecewisePolyLie<Group>(
      std::forward<T1>(x),
      std::move(coefLists),
      std::forward<T2>(poses));
  }

protected:
  template<typename T1, typename T2, typename T3, typename T4>
  static matrix_t generateCoeffsLie(
    const Eigen::DenseBase<T1> & x,
    const Eigen::DenseBase<T2> & omega,
    const Eigen::DenseBase<T3> & v,
    const Eigen::DenseBase<T4> & vNp1)
  {
    static_assert(
      T1::IsVectorAtCompileTime &&
      T2::IsVectorAtCompileTime &&
      T3::IsVectorAtCompileTime &&
      T4::IsVectorAtCompileTime,
      "x, omega, v and vNp1 must be vectors.");

    if (x.size() != (omega.size() + 1)) {
      throw std::invalid_argument("Size of x must be equal to size of omega + 1.");
    }

    if (x.size() != (vNp1.size() + 1)) {
      throw std::invalid_argument("Size of x must be equal to size of vNp1 + 1.");
    }

    if (x.size() != v.size()) {
      throw std::invalid_argument("x and v must have the same size.");
    }

    std::size_t nj = x.size() - 1;
    matrix_t coeffs(4, nj);

    for (std::size_t i = 0; i < nj; i++) {
      const double dx = x[i + 1] - x[i];
      coeffs(3, i) = 0.;
      coeffs(2, i) = v[i];
      coeffs(1, i) = (3. * omega[i] - dx * (2. * v[i] + vNp1[i])) / (dx * dx);
      coeffs(0, i) = (-2. * omega[i] + dx * (v[i] + vNp1[i])) / (dx * dx * dx);
    }

    return coeffs;
  }

  template<typename T1, typename T2>
  static matrix_t generateCoeffs(
    const Eigen::DenseBase<T1> & x,
    const Eigen::DenseBase<T2> & y)
  {
    if (x.size() != y.cols()) {
      throw std::invalid_argument(
              "The number of columns of each element of ys must be equal to the size of x.");
    }

    matrix_t coeffs;

    const std::size_t nx = x.size();
    const std::size_t nj = nx - 1;
    const std::size_t ni = y.rows();

    if (ni == 1) {
      if (nx == 2) {  // If only 2 points, linear interpolation
        coeffs.resize(2, 1);
        coeffs(1, 0) = y(0, 0);
        coeffs(0, 0) = (y(0, 1) - y(0, 0)) / (x[1] - x[0]);

      } else if (nx == 3) {  // If only 3 points, quadratic interpolation
        coeffs.resize(3, 1);

        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();

        A(0, 2) = 1.;

        A(1, 0) = powFast<2>(x[1] - x[0]);
        A(1, 1) = (x[1] - x[0]);
        A(1, 2) = 1.;

        A(2, 0) = powFast<2>(x[2] - x[0]);
        A(2, 1) = (x[2] - x[0]);
        A(2, 2) = 1.;

        b[0] = y(0, 0);
        b[1] = y(0, 1);
        b[2] = y(0, 2);

        Eigen::FullPivLU<Eigen::Matrix3d> dec(A);
        coeffs = dec.solve(b);

      } else {  // If more than 3 points, piecewise cubic interpolation
        coeffs.resize(4, nj);
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4. * nj, 4. * nj);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(4. * nj);

        // Function continuity
        uint64_t i0 = 0;
        for (uint64_t i = 0; i < nx - 1; i++) {
          A(2 * i, 4 * i) = powFast<3>(x[i + 1] - x[i]);
          A(2 * i, 4 * i + 1) = powFast<2>(x[i + 1] - x[i]);
          A(2 * i, 4 * i + 2) = (x[i + 1] - x[i]);
          A(2 * i, 4 * i + 3) = 1.;
          A(2 * i + 1, 4 * i + 3) = 1.;

          b[2 * i] = y(0, i + 1);
          b[2 * i + 1] = y(0, i);
        }

        // First derivative continuity
        i0 += 2 * (nx - 1);
        for (uint64_t i = 0; i < nx - 2; i++) {
          A(i0 + i, 4 * i) = 3. * powFast<2>(x[i + 1] - x[i]);
          A(i0 + i, 4 * i + 1) = 2. * (x[i + 1] - x[i]);
          A(i0 + i, 4 * i + 2) = 1.;
          A(i0 + i, 4 * i + 6) = -1.;
        }

        // Second derivatives continuity
        i0 += (nx - 2);
        for (uint64_t i = 0; i < nx - 2; i++) {
          A(i0 + i, 4 * i) = 6. * (x[i + 1] - x[i]);
          A(i0 + i, 4 * i + 1) = 2.;
          A(i0 + i, 4 * i + 5) = -2.;
        }

        // Not-a-knot conditions
        i0 += (nx - 2);
        A(i0, 0) = 6.;
        A(i0, 4) = -6.;
        A(i0 + 1, 4 * (nx - 3)) = 6.;
        A(i0 + 1, 4 * (nx - 3) + 4) = -6.;

        // Solve linear system
        Eigen::FullPivLU<Eigen::MatrixXd> dec(A);
        Eigen::Map<Eigen::VectorXd> coeffsMap(coeffs.data(), 4 * nj);
        coeffsMap = dec.solve(b);
      }

    } else if (ni == 2) {
      coeffs.resize(4, nj);
      for (uint64_t i = 0; i < nj; i++) {
        Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
        Eigen::Vector4d b = Eigen::Vector4d::Zero();

        // Function continuity
        A(0, 0) = powFast<3>((x[i + 1] - x[i]));
        A(0, 1) = powFast<2>((x[i + 1] - x[i]));
        A(0, 2) = (x[i + 1] - x[i]);
        A(0, 3) = 1.;
        A(1, 3) = 1.;
        b[0] = y(0, i + 1);
        b[1] = y(0, i);

        // First derivatives continuity
        A(2, 0) = 3. * powFast<2>((x[i + 1] - x[i]));
        A(2, 1) = 2. * (x[i + 1] - x[i]);
        A(2, 2) = 1.;
        A(3, 2) = 1.;
        b[2] = y(1, i + 1);
        b[3] = y(1, i);

        // Solve linear system
        Eigen::FullPivLU<Eigen::MatrixXd> dec(A);
        coeffs.col(i) = dec.solve(b);
      }

    } else if (ni == 3) {
      coeffs.resize(6, nj);
      for (uint64_t i = 0; i < nj; i++) {
        Eigen::Matrix<double, 6, 6> A = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

        // Function continuity
        A(0, 0) = powFast<5>((x[i + 1] - x[i]));
        A(0, 1) = powFast<4>((x[i + 1] - x[i]));
        A(0, 2) = powFast<3>((x[i + 1] - x[i]));
        A(0, 3) = powFast<2>((x[i + 1] - x[i]));
        A(0, 4) = (x[i + 1] - x[i]);
        A(0, 5) = 1.;
        A(1, 5) = 1.;
        b[0] = y(0, i + 1);
        b[1] = y(0, i);

        // First derivatives continuity
        A(2, 0) = 5. * powFast<4>((x[i + 1] - x[i]));
        A(2, 1) = 4. * powFast<3>((x[i + 1] - x[i]));
        A(2, 2) = 3. * powFast<2>((x[i + 1] - x[i]));
        A(2, 3) = 2. * (x[i + 1] - x[i]);
        A(2, 4) = 1.;
        A(3, 4) = 1.;
        b[2] = y(1, i + 1);
        b[3] = y(1, i);

        // Second derivatives continuity
        A(4, 0) = 20. * powFast<3>((x[i + 1] - x[i]));
        A(4, 1) = 12. * powFast<2>((x[i + 1] - x[i]));
        A(4, 2) = 6. * (x[i + 1] - x[i]);
        A(4, 3) = 2.;
        A(5, 3) = 2.;
        b[4] = y(2, i + 1);
        b[5] = y(2, i);

        // Solve linear system
        Eigen::FullPivLU<Eigen::Matrix<double, 6, 6>> dec(A);
        coeffs.col(i) = dec.solve(b);
      }

    } else {
      throw std::invalid_argument("y as an unsupported number of rows. Should never happen!");
    }
    return coeffs;
  }
};

}  // namespace cbr

#endif  // CBR_MATH__INTERP__SPLINE_HPP_
