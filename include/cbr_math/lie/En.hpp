// Copyright 2020 Yamaha Motor Corporation, USA

#ifndef CBR_MATH__LIE__EN_HPP_
#define CBR_MATH__LIE__EN_HPP_

#include <Eigen/Core>

#include <tuple>
#include <utility>

namespace cbr::lie
{

template<typename Scalar_, int Dim_, int Options = 0>
class En;

template<typename Scalar_, int Options = 0>
using E1 = En<Scalar_, 1, Options>;
template<typename Scalar_, int Options = 0>
using E2 = En<Scalar_, 2, Options>;
template<typename Scalar_, int Options = 0>
using E3 = En<Scalar_, 3, Options>;
template<typename Scalar_, int Options = 0>
using E4 = En<Scalar_, 4, Options>;
template<typename Scalar_, int Options = 0>
using E5 = En<Scalar_, 5, Options>;
template<typename Scalar_, int Options = 0>
using E6 = En<Scalar_, 6, Options>;
template<typename Scalar_, int Options = 0>
using E7 = En<Scalar_, 7, Options>;
template<typename Scalar_, int Options = 0>
using E8 = En<Scalar_, 8, Options>;
template<typename Scalar_, int Options = 0>
using E9 = En<Scalar_, 9, Options>;

using E1f = E1<float, 0>;
using E2f = E2<float, 0>;
using E3f = E3<float, 0>;
using E4f = E4<float, 0>;
using E5f = E5<float, 0>;
using E6f = E6<float, 0>;
using E7f = E7<float, 0>;
using E8f = E8<float, 0>;
using E9f = E9<float, 0>;
using E1d = E1<double, 0>;
using E2d = E2<double, 0>;
using E3d = E3<double, 0>;
using E4d = E4<double, 0>;
using E5d = E5<double, 0>;
using E6d = E6<double, 0>;
using E7d = E7<double, 0>;
using E8d = E8<double, 0>;
using E9d = E9<double, 0>;

}  // namespace cbr::lie

namespace Eigen::internal
{

template<class Scalar_, int Dim_, int Options_>
struct traits<cbr::lie::En<Scalar_, Dim_, Options_>>
{
  static constexpr int Options = Options_;
  using Scalar = Scalar_;
  static constexpr int Dim = Dim_;
};

template<class Scalar_, int Dim_, int Options_>
struct traits<Map<cbr::lie::En<Scalar_, Dim_>, Options_>>
{
  static constexpr int Options = Options_;
  using Scalar = Scalar_;
  static constexpr int Dim = Dim_;
};

template<class Scalar_, int Dim_, int Options_>
struct traits<Map<const cbr::lie::En<Scalar_, Dim_>, Options_>>
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
class EnBase
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
  Derived & operator=(const EnBase<OtherDerived> & other)  //NOLINT
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
  En<NewScalar, Dim> cast() const
  {
    return En<NewScalar, Dim>(
      static_cast<const Derived &>(*this).translation().template cast<NewScalar>());
  }

  Eigen::Matrix<Scalar, Dim, Dim> Dx_this_mul_exp_x_at_0() const
  {
    return Eigen::Matrix<Scalar, Dim, Dim>::Identity();
  }

  En<Scalar, Dim> inverse() const
  {
    return En<Scalar, Dim>(-static_cast<const Derived &>(*this).translation());
  }

  Tangent log() const
  {
    return static_cast<const Derived &>(*this).translation();
  }

  template<typename OtherDerived>
  En<Scalar, Dim>
  operator*(const EnBase<OtherDerived> & other) const
  {
    return En<Scalar, Dim>(
      static_cast<const Derived &>(*this).translation() +
      static_cast<const OtherDerived &>(other).translation());
  }

  template<typename OtherDerived,
    typename = typename std::enable_if<
      std::is_same<Scalar, ReturnScalar<OtherDerived>>::value
    >::type
  >
  Derived & operator*=(const EnBase<OtherDerived> & other)
  {
    static_cast<Derived &>(*this) = *this * other;
    return static_cast<Derived &>(*this);
  }

protected:
  EnBase() = default;  // don't instantiate CRTP base
};


// STORAGE IMPLEMENTATION
template<typename Scalar, int Dim, int Options>
class En : public EnBase<En<Scalar, Dim, Options>>
{
  using Base = EnBase<En<Scalar, Dim, Options>>;

public:
  using Tangent = typename Base::Tangent;

  using Base::operator=;

  // default initializer
  En()
  {
    data_.setZero();
  }

  // copy constructor (allow implicit conversion)
  template<typename OtherDerived>
  explicit En(const EnBase<OtherDerived> & other)
  : data_(static_cast<const OtherDerived &>(other).translation())
  {}

  // forwarding constructor (protected from overriding constructor above)
  template<typename ... S,
    typename = std::enable_if_t<
      sizeof...(S) != 1 ||
      (
        (!std::is_base_of_v<EnBase<std::decay_t<S>>, std::decay_t<S>>&& ...) &&
        (!std::is_base_of_v<En<Scalar, Dim, Options>, std::decay_t<S>>&& ...)
      )
    >
  >
  explicit En(S && ... data)
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

  static En<Scalar, Dim, Options> exp(const Tangent & v)
  {
    return En<Scalar, Dim, Options>(v);
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
class Map<cbr::lie::En<Scalar, Dim, 0>, Options>
  : public cbr::lie::EnBase<Map<cbr::lie::En<Scalar, Dim, 0>, Options>>
{
  using Base = cbr::lie::EnBase<Map<cbr::lie::En<Scalar, Dim, 0>, Options>>;

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
class Map<const cbr::lie::En<Scalar, Dim, 0>, Options>
  : public cbr::lie::EnBase<Map<const cbr::lie::En<Scalar, Dim, 0>, Options>>
{
  using Base = cbr::lie::EnBase<Map<const cbr::lie::En<Scalar, Dim, 0>, Options>>;

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

#endif   // CBR_MATH__LIE__EN_HPP_
