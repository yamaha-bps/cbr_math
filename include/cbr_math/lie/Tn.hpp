// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_MATH__LIE__TN_HPP_
#define CBR_MATH__LIE__TN_HPP_

#include <Eigen/Core>

#include <tuple>
#include <utility>

namespace cbr::lie
{

template<typename Scalar_, int Dim_, int Options = 0>
class Tn;

template<typename Scalar_, int Options = 0>
using T1 = Tn<Scalar_, 1, Options>;
template<typename Scalar_, int Options = 0>
using T2 = Tn<Scalar_, 2, Options>;
template<typename Scalar_, int Options = 0>
using T3 = Tn<Scalar_, 3, Options>;
template<typename Scalar_, int Options = 0>
using T4 = Tn<Scalar_, 4, Options>;
template<typename Scalar_, int Options = 0>
using T5 = Tn<Scalar_, 5, Options>;
template<typename Scalar_, int Options = 0>
using T6 = Tn<Scalar_, 6, Options>;
template<typename Scalar_, int Options = 0>
using T7 = Tn<Scalar_, 7, Options>;
template<typename Scalar_, int Options = 0>
using T8 = Tn<Scalar_, 8, Options>;
template<typename Scalar_, int Options = 0>
using T9 = Tn<Scalar_, 9, Options>;

using T1f = T1<float, 0>;
using T2f = T2<float, 0>;
using T3f = T3<float, 0>;
using T4f = T4<float, 0>;
using T5f = T5<float, 0>;
using T6f = T6<float, 0>;
using T7f = T7<float, 0>;
using T8f = T8<float, 0>;
using T9f = T9<float, 0>;
using T1d = T1<double, 0>;
using T2d = T2<double, 0>;
using T3d = T3<double, 0>;
using T4d = T4<double, 0>;
using T5d = T5<double, 0>;
using T6d = T6<double, 0>;
using T7d = T7<double, 0>;
using T8d = T8<double, 0>;
using T9d = T9<double, 0>;

}  // namespace cbr::lie

namespace Eigen::internal
{

template<class Scalar_, int Dim_, int Options_>
struct traits<cbr::lie::Tn<Scalar_, Dim_, Options_>>
{
  static constexpr int Options = Options_;
  using Scalar = Scalar_;
  static constexpr int Dim = Dim_;
};

template<class Scalar_, int Dim_, int Options_>
struct traits<Map<cbr::lie::Tn<Scalar_, Dim_>, Options_>>
{
  static constexpr int Options = Options_;
  using Scalar = Scalar_;
  static constexpr int Dim = Dim_;
};

template<class Scalar_, int Dim_, int Options_>
struct traits<Map<const cbr::lie::Tn<Scalar_, Dim_>, Options_>>
{
  static constexpr int Options = Options_;
  using Scalar = Scalar_;
  static constexpr int Dim = Dim_;
};

}  // namespace Eigen::internal


namespace cbr::lie
{

// BASE
template<class Derived>
class TnBase
{
public:
  static constexpr int Dim = Eigen::internal::traits<Derived>::Dim;
  static constexpr int DoF = Dim;
  static constexpr int num_parameters = Dim;

  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;
  using Tangent = Eigen::Matrix<Scalar, Dim, 1>;
  using Adjoint = Eigen::Matrix<Scalar, Dim, Dim>;

  template<typename OtherDerived>
  using ReturnScalar =
    typename Eigen::ScalarBinaryOpTraits<Scalar, typename OtherDerived::Scalar>::ReturnType;

  // copy assignment from OtherDerived
  template<typename OtherDerived>
  Derived & operator=(const TnBase<OtherDerived> & other)  //NOLINT
  {
    static_cast<Derived &>(*this).translation() =
      static_cast<const OtherDerived &>(other).translation();
    return static_cast<Derived &>(*this);
  }

  // copy assignment from Eigen matrix
  template<typename OtherDerived>
  Derived & operator=(const Eigen::EigenBase<OtherDerived> & mat)  //NOLINT
  {
    static_cast<Derived &>(*this).translation() = mat;
    return static_cast<Derived &>(*this);
  }

  Adjoint Adj() const
  {
    return Adjoint::Identity();
  }

  template<typename NewScalar>
  Tn<NewScalar, Dim> cast() const
  {
    return Tn<NewScalar, Dim>(
      static_cast<const Derived &>(*this).translation().template cast<NewScalar>());
  }

  Eigen::Matrix<Scalar, Dim, Dim> Dx_this_mul_exp_x_at_0() const
  {
    return Eigen::Matrix<Scalar, Dim, Dim>::Identity();
  }

  Tn<Scalar, Dim> inverse() const
  {
    return Tn<Scalar, Dim>(-static_cast<const Derived &>(*this).translation());
  }

  Tangent log() const
  {
    return static_cast<const Derived &>(*this).translation();
  }

  template<typename OtherDerived>
  Tn<Scalar, Dim>
  operator*(const TnBase<OtherDerived> & other) const
  {
    return Tn<Scalar, Dim>(
      static_cast<const Derived &>(*this).translation() +
      static_cast<const OtherDerived &>(other).translation());
  }

  template<typename OtherDerived,
    typename = typename std::enable_if<
      std::is_same<Scalar, ReturnScalar<OtherDerived>>::value
    >::type
  >
  Derived & operator*=(const TnBase<OtherDerived> & other)
  {
    static_cast<Derived &>(*this) = *this * other;
    return static_cast<Derived &>(*this);
  }

protected:
  TnBase() = default;  // don't instantiate CRTP base
};


// STORAGE IMPLEMENTATION
template<typename Scalar, int Dim, int Options>
class Tn : public TnBase<Tn<Scalar, Dim, Options>>
{
  using Base = TnBase<Tn<Scalar, Dim, Options>>;

public:
  using Tangent = typename Base::Tangent;

  using Base::operator=;

  // default initializer
  Tn()
  {
    data_.setZero();
  }

  // copy constructor (allow implicit conversion)
  template<typename OtherDerived>
  explicit Tn(const TnBase<OtherDerived> & other)
  : data_(static_cast<const OtherDerived &>(other).translation())
  {}

  // forwarding constructor (protected from overriding constructor above)
  template<typename ... S,
    typename = std::enable_if_t<
      sizeof...(S) != 1 ||
      (
        (!std::is_base_of_v<TnBase<std::decay_t<S>>, std::decay_t<S>>&& ...) &&
        (!std::is_base_of_v<Tn<Scalar, Dim, Options>, std::decay_t<S>>&& ...)
      )
    >
  >
  explicit Tn(S && ... data)
  : data_(std::forward<S>(data) ...)
  {}

  Eigen::Matrix<Scalar, Dim, 1, Options> & translation()
  {
    return data_;
  }

  const Eigen::Matrix<Scalar, Dim, 1, Options> & translation() const
  {
    return data_;
  }

  Scalar * data()
  {
    return data_.data();
  }

  static Tn<Scalar, Dim, Options> exp(const Tangent & v)
  {
    return Tn<Scalar, Dim, Options>(v);
  }

  static Tangent lieBracket(const Tangent &, const Tangent &)
  {
    return Tangent::Zero();
  }

protected:
  Eigen::Matrix<Scalar, Dim, 1, Options> data_;
};

}  // namespace cbr::lie


namespace Eigen
{

// Overload Eigen::Map<En>
template<typename Scalar, int Dim, int Options>
class Map<cbr::lie::Tn<Scalar, Dim, 0>, Options>
  : public cbr::lie::TnBase<Map<cbr::lie::Tn<Scalar, Dim, 0>, Options>>
{
  using Base = cbr::lie::TnBase<Map<cbr::lie::Tn<Scalar, Dim, 0>, Options>>;

public:
  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  explicit Map(Scalar * x)
  : map_(x) {}

  Eigen::Map<Eigen::Matrix<Scalar, Dim, 1>, Options> & translation()
  {
    return map_;
  }

  const Eigen::Map<Eigen::Matrix<Scalar, Dim, 1>, Options> & translation() const
  {
    return map_;
  }

private:
  Eigen::Map<Eigen::Matrix<Scalar, Dim, 1>, Options> map_;
};


// Overload Eigen::Map<En>
template<typename Scalar, int Dim, int Options>
class Map<const cbr::lie::Tn<Scalar, Dim, 0>, Options>
  : public cbr::lie::TnBase<Map<const cbr::lie::Tn<Scalar, Dim, 0>, Options>>
{
  using Base = cbr::lie::TnBase<Map<const cbr::lie::Tn<Scalar, Dim, 0>, Options>>;

public:
  using Base::operator*;

  explicit Map(const Scalar * x)
  : map_(x) {}

  const Eigen::Map<const Eigen::Matrix<Scalar, Dim, 1>, Options> & translation() const
  {
    return map_;
  }

private:
  const Eigen::Map<const Eigen::Matrix<Scalar, Dim, 1>, Options> map_;
};

}  // namespace Eigen

#endif   // CBR_MATH__LIE__TN_HPP_
