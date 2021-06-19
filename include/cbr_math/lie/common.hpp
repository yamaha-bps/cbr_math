// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE


#ifndef CBR_MATH__LIE__COMMON_HPP_
#define CBR_MATH__LIE__COMMON_HPP_

#include <Eigen/Core>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <boost/math/special_functions/bernoulli.hpp>
#include <boost/math/special_functions/factorials.hpp>

#include <limits>

#include "Tn.hpp"
#include "group_product.hpp"


namespace cbr::lie
{

namespace detail
{
template<typename T>
struct dexp_trait
{};

/**
 * @brief Calculate dr_expinv_x for T(n)
 */
template<typename Scalar, int Dim>
struct dexp_trait<Tn<Scalar, Dim>>
{
  using Jacobian = Eigen::Matrix<Scalar, Dim, Dim>;

  template<typename Derived>
  static Jacobian calc(const Eigen::MatrixBase<Derived> &)
  {
    return Jacobian::Identity();
  }

  template<typename Derived>
  static Jacobian calc_inv(const Eigen::MatrixBase<Derived> &)
  {
    return Jacobian::Identity();
  }
};

/**
 * @brief Calculate dr_exp_x / dr_expinv_x for SE(2)
 */
template<typename Scalar>
struct dexp_trait<Sophus::SE2<Scalar>>
{
  static constexpr auto DoF = 3;

  using Tangent = typename Sophus::SE2<Scalar>::Tangent;
  using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;

  template<typename Derived>
  static Jacobian calc(const Eigen::MatrixBase<Derived> & x)
  {
    using std::cos, std::sin;

    const Scalar th_sq = x(2) * x(2);

    Eigen::Matrix<Scalar, 3, 3> W;
    W.setZero();
    W.template topLeftCorner<2, 2>() = Sophus::SO2<Scalar>::hat(x(2));
    W(0, 2) = x(1);
    W(1, 2) = -x(0);

    // Small angle approximation
    if (th_sq <= 1e-10) {
      return Jacobian::Identity() - Scalar(0.5) * W;
    }

    const Scalar th = abs(x(2));

    return Jacobian::Identity() - (1 - cos(th)) / th_sq * W + (th - sin(th)) / (th_sq * th) * W * W;
  }

  template<typename Derived>
  static Jacobian calc_inv(const Eigen::MatrixBase<Derived> & x)
  {
    using std::cos, std::sin;

    const Scalar th_sq = x(2) * x(2);

    Eigen::Matrix<Scalar, 3, 3> W;
    W.setZero();
    W.template topLeftCorner<2, 2>() = Sophus::SO2<Scalar>::hat(x(2));
    W(0, 2) = x(1);
    W(1, 2) = -x(0);

    // Small angle approximation
    if (th_sq <= 1e-10) {
      return Jacobian::Identity() + Scalar(0.5) * W;
    }

    const Scalar th = abs(x(2));
    const Scalar k = Scalar(1) / th_sq - (Scalar(1) + cos(th)) / (Scalar(2) * th * sin(th));

    return Jacobian::Identity() + Scalar(0.5) * W + k * W * W;
  }
};

/**
 * @brief Calculate dr_exp_x / dr_expinv_x for SO(3)
 */
template<typename Scalar>
struct dexp_trait<Sophus::SO3<Scalar>>
{
  static constexpr auto DoF = 3;

  using Tangent = typename Sophus::SO3<Scalar>::Tangent;
  using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;

  template<typename Derived>
  static Jacobian calc(const Eigen::MatrixBase<Derived> & x)
  {
    using std::cos, std::sin, std::sqrt;

    const Scalar th_sq = x.squaredNorm();

    const auto W = Sophus::SO3<Scalar>::hat(x);

    // Small angle approximation
    if (th_sq <= 1e-10) {
      return Jacobian::Identity() - Scalar(0.5) * W;
    }

    const Scalar th = sqrt(th_sq);
    return Jacobian::Identity() -
           (Scalar(1) - cos(th)) / th_sq * W +
           (th - sin(th)) / (th_sq * th) * W * W;
  }

  template<typename Derived>
  static Jacobian calc_inv(const Eigen::MatrixBase<Derived> & x)
  {
    using std::cos, std::sin, std::sqrt;

    const Scalar th_sq = x.squaredNorm();

    const auto W = Sophus::SO3<Scalar>::hat(x);

    // Small angle approximation
    if (th_sq <= 1e-10) {
      return Jacobian::Identity() + Scalar(0.5) * W;
    }

    const Scalar th = sqrt(th_sq);
    const Scalar k = Scalar(1) / th_sq - (Scalar(1) + cos(th)) / (Scalar(2) * th * sin(th));

    return Jacobian::Identity() + Scalar(0.5) * W + k * W * W;
  }
};


/**
 * @brief Calculate dr_exp_x / dr_expinv_x for SE(3)
 */
template<typename Scalar>
struct dexp_trait<Sophus::SE3<Scalar>>
{
  static constexpr auto DoF = 6;

  using Tangent = typename Sophus::SE3<Scalar>::Tangent;
  using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;

  template<typename Derived>
  static Jacobian calc(const Eigen::MatrixBase<Derived> & x)
  {
    using std::cos, std::sin, std::sqrt;

    Jacobian ret;
    ret.setZero();

    Eigen::Matrix<Scalar, 3, 3> J =
      dexp_trait<Sophus::SO3<Scalar>>::calc(x.template tail<3>());

    ret.template topLeftCorner<3, 3>() = J;
    ret.template bottomRightCorner<3, 3>() = J;

    const Scalar th_sq = x.template tail<3>().squaredNorm();

    // negative because it's the right one
    Eigen::Matrix<Scalar, 3, 3> V = Sophus::SO3<Scalar>::hat(-x.template head<3>());
    Eigen::Matrix<Scalar, 3, 3> W = Sophus::SO3<Scalar>::hat(-x.template tail<3>());

    Eigen::Matrix<Scalar, 3, 3> Q = Scalar(0.5) * V;

    if (th_sq < 1e-10) {
      // small angle approx
      Q += Scalar(1) / Scalar(6) * (W * V + V * W);
    } else {
      // pre-calc some quantities that are used multiple times
      const Scalar th = sqrt(th_sq);
      const Scalar th_4 = th_sq * th_sq;
      const Scalar vdw = x.template tail<3>().dot(x.template head<3>());
      const Eigen::Matrix<Scalar, 3, 3> WV = W * V, VW = V * W, WW = W * W;
      const Scalar cTh = cos(th);
      const Scalar sTh = sin(th);

      Q += (th - sTh) / (th * th_sq) * (WV + VW - vdw * W);
      Q += (cTh - Scalar(1) + th_sq / Scalar(2)) / th_4 *
        (W * WV + VW * W + vdw * (Scalar(3) * W - WW));
      Q -= Scalar(3) * vdw * (th - sTh - th * th_sq / Scalar(6)) / (th_4 * th) * WW;
    }

    ret.template topRightCorner<3, 3>() = Q;

    return ret;
  }

  template<typename Derived>
  static Jacobian calc_inv(const Eigen::MatrixBase<Derived> & x)
  {
    using std::cos, std::sin, std::sqrt;

    Jacobian ret;
    ret.setZero();

    Eigen::Matrix<Scalar, 3, 3> Jinv =
      dexp_trait<Sophus::SO3<Scalar>>::calc_inv(x.template tail<3>());

    ret.template topLeftCorner<3, 3>() = Jinv;
    ret.template bottomRightCorner<3, 3>() = Jinv;

    const Scalar th_sq = x.template tail<3>().squaredNorm();

    // negative because it's the right one
    Eigen::Matrix<Scalar, 3, 3> V = Sophus::SO3<Scalar>::hat(-x.template head<3>());
    Eigen::Matrix<Scalar, 3, 3> W = Sophus::SO3<Scalar>::hat(-x.template tail<3>());

    Eigen::Matrix<Scalar, 3, 3> Q = Scalar(0.5) * V;

    if (th_sq < 1e-10) {
      // small angle approx
      Q += Scalar(1) / Scalar(6) * (W * V + V * W);
    } else {
      // pre-calc some quantities that are used multiple times
      const Scalar th = sqrt(th_sq);
      const Scalar th_4 = th_sq * th_sq;
      const Scalar vdw = x.template tail<3>().dot(x.template head<3>());
      const Eigen::Matrix<Scalar, 3, 3> WV = W * V, VW = V * W, WW = W * W;
      const Scalar cTh = cos(th);
      const Scalar sTh = sin(th);

      Q += (th - sTh) / (th * th_sq) * (WV + VW - vdw * W);
      Q += (cTh - Scalar(1) + th_sq / Scalar(2)) / th_4 *
        (W * WV + VW * W + vdw * (Scalar(3) * W - WW));
      Q -= Scalar(3) * vdw * (th - sTh - th * th_sq / Scalar(6)) / (th_4 * th) * WW;
    }

    ret.template topRightCorner<3, 3>() = -Jinv * Q * Jinv;

    return ret;
  }
};


/**
 * @brief Calculate dr_exp_x / dr_expinv_x for group product
 */
template<typename Scalar, template<typename, int> typename ... T>
struct dexp_trait<GroupProduct<Scalar, 0, T...>>
{
  using lie_t = GroupProduct<Scalar, 0, T...>;
  static constexpr auto DoF = lie_t::DoF;

  using Tangent = typename lie_t::Tangent;
  using Jacobian = Eigen::Matrix<Scalar, DoF, DoF>;

  template<int ... Idx, int ... TBeg, int ... TLen>
  static Jacobian calc_impl(
    const Tangent & x,
    std::integer_sequence<int, Idx...>,
    std::integer_sequence<int, TBeg...>,
    std::integer_sequence<int, TLen...>
  )
  {
    Jacobian res;
    res.setZero();
    (
      (
        res.template block<TLen, TLen>(TBeg, TBeg) =
        dexp_trait<typename lie_t::template type_t<Idx>>::template calc(
          x.template segment<TLen>(TBeg)
        )
      ),
      ...
    );
    return res;
  }

  template<int ... Idx, int ... TBeg, int ... TLen>
  static Jacobian calc_inv_impl(
    const Tangent & x,
    std::integer_sequence<int, Idx...>,
    std::integer_sequence<int, TBeg...>,
    std::integer_sequence<int, TLen...>
  )
  {
    Jacobian res;
    res.setZero();
    (
      (
        res.template block<TLen, TLen>(TBeg, TBeg) =
        dexp_trait<typename lie_t::template type_t<Idx>>::template calc_inv(
          x.template segment<TLen>(TBeg)
        )
      ),
      ...
    );
    return res;
  }

  template<typename Derived>
  static Jacobian calc(const Eigen::MatrixBase<Derived> & x)
  {
    return calc_impl(
      x, typename lie_t::idx_t{}, typename lie_t::tidx_beg_t{},
      typename lie_t::tidx_len_t{});
  }

  template<typename Derived>
  static Jacobian calc_inv(const Eigen::MatrixBase<Derived> & x)
  {
    return calc_inv_impl(
      x, typename lie_t::idx_t{}, typename lie_t::tidx_beg_t{},
      typename lie_t::tidx_len_t{});
  }
};

}  // namespace detail

/**
 * @brief Calculate dr_exp_x using closed-loop formulas
 */
template<typename T>
auto dr_exp([[maybe_unused]] const typename T::Tangent x)
{
  if constexpr (std::is_base_of_v<Sophus::SO2Base<T>, T>) {
    return typename T::Scalar(1);
  } else {
    return detail::dexp_trait<T>::calc(x);
  }
}

/**
 * @brief Calculate dr_expinv_x using closed-loop formulas
 */
template<typename T>
auto dr_expinv([[maybe_unused]] const typename T::Tangent x)
{
  if constexpr (std::is_base_of_v<Sophus::SO2Base<T>, T>) {
    return typename T::Scalar(1);
  } else {
    return detail::dexp_trait<T>::calc_inv(x);
  }
}

/**
 * Calculate a r-order approximation to dexpinv(u, v) = \sum_k (B_k / k!) ad_u^k v
 * where B_k is the k:th Bernoulli number and ad_u^k v = [u, [u, ..., [u, v]]]
 */
template<typename T>
typename T::Tangent dl_expinv_approx(
  const typename T::Tangent u, const typename T::Tangent & v, uint32_t r)
{
  using Scalar = typename T::Scalar;
  typename T::Tangent ret = v;
  typename T::Tangent tmp = v;

  if (r > 2) {
    r = r - (r % 2);
  }

  for (auto k = 1u; k <= r; ++k) {
    tmp = T::lieBracket(u, tmp);
    if (k == 1) {
      ret -= 0.5 * tmp;
    } else if (k % 2 == 0) {
      ret += (boost::math::bernoulli_b2n<Scalar>(k / 2) / boost::math::factorial<Scalar>(k)) * tmp;
    }
  }

  return ret;
}

namespace detail
{

// Cast an Eigen type T<U> to a target scalar type T<S>
template<typename T, typename S>
struct change_scalar_eigen
{
  using type = std::remove_const_t<typename T::template CastXpr<S>::Type::EvalReturnType>;
};

// Partial specialization when S == T::Scalar
template<typename T>
struct change_scalar_eigen<T, typename T::Scalar>
{
  using type = T;
};


template<typename T, typename Scalar, typename = std::void_t<>>
struct change_scalar;

// Lie group traits default: regular eigen type
template<typename T, typename Scalar>
struct change_scalar<T, Scalar, std::enable_if_t<std::is_base_of_v<Eigen::EigenBase<T>, T>, void>>
{
  using type = typename change_scalar_eigen<T, Scalar>::type;
};

// Specialization valid when T::Tangent exists (as in Sophus Lie groups)
template<typename T, typename Scalar>
struct change_scalar<T, Scalar, std::void_t<typename T::Tangent>>
{
  using type = typename std::result_of_t<decltype(&T::template cast<Scalar>)(T)>;
};

template<typename T, typename Scalar>
using change_scalar_t = typename change_scalar<T, Scalar>::type;

}  // namespace detail

}  // namespace cbr::lie

#endif  // CBR_MATH__LIE__COMMON_HPP_
