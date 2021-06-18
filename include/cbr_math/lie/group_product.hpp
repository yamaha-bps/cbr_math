// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE


#ifndef CBR_MATH__LIE__GROUP_PRODUCT_HPP_
#define CBR_MATH__LIE__GROUP_PRODUCT_HPP_

#include <Eigen/Core>

#include <tuple>
#include <utility>

namespace cbr::lie
{

template<typename Scalar, int Options = 0, template<typename, int> typename ... T>
class GroupProduct;

// Alias with Options = 0
template<typename Scalar, template<typename, int> typename ... T>
using GroupProductDef = GroupProduct<Scalar, 0, T...>;

}  // namespace cbr::lie

namespace Eigen::internal
{

// Cumulative prefix sum for std::integer_sequence (c.f. std::exclusive_scan)
template<typename Cur, typename ISeq, typename ISeq::value_type Sum>
struct iseq_psum
{
  using type = Cur;
};

template<typename T, T... Cur, T First, T... Rem, T Sum>
struct iseq_psum<std::integer_sequence<T, Cur...>, std::integer_sequence<T, First, Rem...>, Sum>
  : iseq_psum<std::integer_sequence<T, Cur..., Sum>, std::integer_sequence<T, Rem...>, Sum + First>
{};

template<class ISeq>
using iseq_psum_t = typename iseq_psum<
  std::integer_sequence<typename ISeq::value_type>, ISeq, 0>::type;

// Sum for std::integer_sequence
template<typename ISeq>
struct iseq_sum;

template<typename T, T ... I>
struct iseq_sum<std::integer_sequence<T, I...>>
{
  static constexpr T value = (I + ...);
};

template<typename ISeq>
constexpr typename ISeq::value_type iseq_sum_v = iseq_sum<ISeq>::value;

template<class Scalar_, int Options_, template<typename, int> typename ... T>
struct traits<cbr::lie::GroupProduct<Scalar_, Options_, T...>>
{
  static constexpr int Options = Options_;
  using Scalar = Scalar_;
  using idx_t = std::make_integer_sequence<int, sizeof...(T)>;
  // size T members as integer_sequence
  using midx_len_t = std::integer_sequence<int, T<Scalar, Options_>::num_parameters ...>;
  // zero index of T members as integer_sequence
  using midx_beg_t = iseq_psum_t<midx_len_t>;
  // size T members as integer_sequence
  using tidx_len_t = std::integer_sequence<int, T<Scalar, Options_>::DoF ...>;
  // size T members as integer_sequence
  using tidx_beg_t = iseq_psum_t<tidx_len_t>;
  static constexpr size_t size = sizeof...(T);
  static constexpr int DoF = iseq_sum_v<tidx_len_t>;
  static constexpr int num_parameters = iseq_sum_v<midx_len_t>;
  template<typename NewScalar>
  using state_t = cbr::lie::GroupProduct<NewScalar, 0, T...>;
  template<size_t Idx>
  using type_t = typename std::tuple_element_t<Idx, std::tuple<T<Scalar, Options_>...>>;
};

template<class Scalar_, int Options_, template<typename, int> typename ... T>
struct traits<Map<cbr::lie::GroupProduct<Scalar_, 0, T...>, Options_>>
  : public traits<cbr::lie::GroupProduct<Scalar_, Options_, T...>>
{
  using Base = traits<cbr::lie::GroupProduct<Scalar_, Options_, T...>>;
  static constexpr int Options = Options_;
  using Scalar = Scalar_;

  using idx_t = typename Base::idx_t;
  using midx_beg_t = typename Base::midx_beg_t;
  using midx_len_t = typename Base::midx_len_t;
  using tidx_beg_t = typename Base::tidx_beg_t;
  using tidx_len_t = typename Base::tidx_len_t;
  static constexpr size_t size = sizeof...(T);
  static constexpr int DoF = Base::DoF;
  static constexpr int num_parameters = Base::num_parameters;
  template<typename NewScalar>
  using state_t = cbr::lie::GroupProduct<NewScalar, 0, T...>;
  template<size_t Idx>
  using type_t = typename std::tuple_element_t<Idx, std::tuple<T<Scalar, Options_>...>>;
};

template<class Scalar_, int Options_, template<typename, int> typename ... T>
struct traits<Map<const cbr::lie::GroupProduct<Scalar_, 0, T...>, Options_>>
  : public traits<const cbr::lie::GroupProduct<Scalar_, Options_, T...>>
{
  using Base = traits<cbr::lie::GroupProduct<Scalar_, Options_, T...>>;
  static constexpr int Options = Options_;
  using Scalar = Scalar_;

  using idx_t = typename Base::idx_t;
  using midx_beg_t = typename Base::midx_beg_t;
  using midx_len_t = typename Base::midx_len_t;
  using tidx_beg_t = typename Base::tidx_beg_t;
  using tidx_len_t = typename Base::tidx_len_t;
  static constexpr size_t size = sizeof...(T);
  static constexpr int DoF = Base::DoF;
  static constexpr int num_parameters = Base::num_parameters;
  template<typename NewScalar>
  using state_t = cbr::lie::GroupProduct<NewScalar, 0, T...>;
  template<size_t Idx>
  using type_t = typename std::tuple_element_t<Idx, std::tuple<T<Scalar, Options_>...>>;
};

}  // namespace Eigen::internal


namespace cbr::lie
{

// BASE
template<class Derived>
class GroupProductBase
{
public:
  using Scalar = typename Eigen::internal::traits<Derived>::Scalar;

  static constexpr size_t size = Eigen::internal::traits<Derived>::size;
  static constexpr int DoF = Eigen::internal::traits<Derived>::DoF;
  static constexpr int num_parameters = Eigen::internal::traits<Derived>::num_parameters;

  using Tangent = Eigen::Matrix<Scalar, DoF, 1>;
  using Adjoint = Eigen::Matrix<Scalar, DoF, DoF>;

  using idx_t = typename Eigen::internal::traits<Derived>::idx_t;
  using midx_beg_t = typename Eigen::internal::traits<Derived>::midx_beg_t;
  using midx_len_t = typename Eigen::internal::traits<Derived>::midx_len_t;
  using tidx_beg_t = typename Eigen::internal::traits<Derived>::tidx_beg_t;
  using tidx_len_t = typename Eigen::internal::traits<Derived>::tidx_len_t;

  template<typename NewScalar>
  using state_t = typename Eigen::internal::traits<Derived>::template state_t<NewScalar>;

  template<int Idx>
  using type_t = typename Eigen::internal::traits<Derived>::template type_t<Idx>;

  template<typename OtherDerived>
  using ReturnScalar =
    typename Eigen::ScalarBinaryOpTraits<Scalar, typename OtherDerived::Scalar>::ReturnType;

  // adjoint matrix
  Adjoint Adj() const
  {
    return Adj_impl(idx_t{}, tidx_beg_t{}, tidx_len_t{});
  }

  // cast to different datatype
  template<typename NewScalar>
  state_t<NewScalar> cast() const
  {
    return this->template cast_impl<NewScalar>(idx_t{});
  }

  // derivative of x * exp(v) at v=0
  Eigen::Matrix<Scalar, num_parameters, DoF> Dx_this_mul_exp_x_at_0() const
  {
    return Dx_this_mul_exp_x_at_0_impl(
      idx_t{}, midx_beg_t{}, midx_len_t{}, tidx_beg_t{}, tidx_len_t{}
    );
  }

  // inverse
  state_t<Scalar> inverse() const
  {
    return inverse_impl(idx_t{});
  }

  // logarithm
  Tangent log() const
  {
    return log_impl(idx_t{}, tidx_beg_t{}, tidx_len_t{});
  }

  // group multiplication
  template<typename OtherDerived>
  state_t<Scalar> operator*(const GroupProductBase<OtherDerived> & other) const
  {
    return mul_impl<OtherDerived>(other, idx_t{});
  }

  // in-place group multiplication
  template<typename OtherDerived,
    typename = typename std::enable_if<
      std::is_same<Scalar, ReturnScalar<OtherDerived>>::value
    >::type
  >
  Derived & operator*=(const GroupProductBase<OtherDerived> & other)
  {
    imul_impl<OtherDerived>(other, idx_t{});
    return static_cast<Derived &>(*this);
  }

  // copy assignment operator from OtherDerived
  template<typename OtherDerived>
  Derived & operator=(const GroupProductBase<OtherDerived> & other)  //NOLINT
  {
    assign_impl(other, idx_t{});
    return static_cast<Derived &>(*this);
  }

protected:
  GroupProductBase() = default;  // don't instantiate CRTP base

private:
  template<typename OtherDerived, int ... Idx>
  void assign_impl(const GroupProductBase<OtherDerived> & other, std::integer_sequence<int, Idx...>)
  {
    ((std::get<Idx>(static_cast<Derived &>(*this)) =
    std::get<Idx>(static_cast<const OtherDerived &>(other))
    ), ...);
  }

  template<int ... Idx, int ... TBeg, int ... TLen>
  Adjoint Adj_impl(
    std::integer_sequence<int, Idx...>,
    std::integer_sequence<int, TBeg...>,
    std::integer_sequence<int, TLen...>) const
  {
    Adjoint res;
    res.setZero();
    ((res.template block<TLen, TLen>(TBeg, TBeg) =
    std::get<Idx>(static_cast<const Derived &>(*this)).Adj()
    ), ...);
    return res;
  }

  template<typename NewScalar, int ... Idx>
  state_t<NewScalar> cast_impl(std::integer_sequence<int, Idx...>) const
  {
    return state_t<NewScalar>(
      std::get<Idx>(static_cast<const Derived &>(*this)).template cast<NewScalar>() ...);
  }

  template<int ... Idx, int ... MBeg, int ... MLen, int ... TBeg, int ... TLen>
  Eigen::Matrix<Scalar, num_parameters, DoF> Dx_this_mul_exp_x_at_0_impl(
    std::integer_sequence<int, Idx...>, std::integer_sequence<int, MBeg...>,
    std::integer_sequence<int, MLen...>, std::integer_sequence<int, TBeg...>,
    std::integer_sequence<int, TLen...>
  ) const
  {
    Eigen::Matrix<Scalar, num_parameters, DoF> res;
    res.setZero();
    ((res.template block<MLen, TLen>(MBeg, TBeg) =
    std::get<Idx>(static_cast<const Derived &>(*this)).Dx_this_mul_exp_x_at_0()
    ), ...);
    return res;
  }

  template<int ... Idx>
  state_t<Scalar> inverse_impl(std::integer_sequence<int, Idx...>) const
  {
    return state_t<Scalar>(std::get<Idx>(static_cast<const Derived &>(*this)).inverse() ...);
  }

  template<int ... Idx, int ... TBeg, int ... TLen>
  Tangent log_impl(
    std::integer_sequence<int, Idx...>, std::integer_sequence<int, TBeg...>,
    std::integer_sequence<int, TLen...>
  ) const
  {
    Tangent res;
    ((res.template segment<TLen>(TBeg) =
    std::get<Idx>(static_cast<const Derived &>(*this)).log()), ...);
    return res;
  }

  template<typename OtherDerived, int ... Idx>
  state_t<ReturnScalar<OtherDerived>> mul_impl(
    const GroupProductBase<OtherDerived> & other, std::integer_sequence<int, Idx...>
  ) const
  {
    return state_t<ReturnScalar<OtherDerived>>(
      (std::get<Idx>(static_cast<const Derived &>(*this)) *
      std::get<Idx>(static_cast<const OtherDerived &>(other))) ...);
  }

  template<typename OtherDerived, int ... Idx>
  void imul_impl(const GroupProductBase<OtherDerived> & other, std::integer_sequence<int, Idx...>)
  {
    ((std::get<Idx>(static_cast<Derived &>(*this)) *=
    std::get<Idx>(static_cast<const OtherDerived &>(other))), ...);
  }
};


// STORAGE IMPLEMENTATION
template<typename Scalar, int Options, template<typename, int> typename ... T>
class GroupProduct
  : public GroupProductBase<GroupProduct<Scalar, Options, T...>>,
  public std::tuple<T<Scalar, Options>...>
{
  using Base = GroupProductBase<GroupProduct<Scalar, Options, T...>>;

public:
  using Tangent = typename Base::Tangent;

  using tidx_beg_t = typename Base::tidx_beg_t;
  using tidx_len_t = typename Base::tidx_len_t;

  using Base::operator=;

  // default initializer
  GroupProduct()
  : Base{},
    std::tuple<T<Scalar, Options>...>{}
  {}

  // copy constructor from OtherDerived
  template<typename OtherDerived>
  explicit GroupProduct(const GroupProductBase<OtherDerived> & other)
  : GroupProduct(other, std::make_integer_sequence<int, Base::size>{})
  {}

  // forwarding constructor (protected from overriding constructor above)
  template<typename ... S,
    typename = std::enable_if_t<
      sizeof...(S) != 1 ||
      (
        (!std::is_base_of_v<GroupProductBase<std::decay_t<S>>, std::decay_t<S>>&& ...) &&
        (!std::is_base_of_v<GroupProduct<Scalar, Options, T...>, std::decay_t<S>>&& ...)
      )
    >
  >
  explicit GroupProduct(S && ... args)
  : std::tuple<T<Scalar, Options>...>(std::forward<S>(args) ...)
  {}

  // group exponential
  static GroupProduct<Scalar, Options, T...> exp(const Tangent & v)
  {
    return exp_impl(v, tidx_beg_t{}, tidx_len_t{});
  }

  // lie bracket
  static Tangent lieBracket(const Tangent & a, const Tangent & b)
  {
    return lieBracket_impl(a, b, tidx_beg_t{}, tidx_len_t{});
  }

private:
  template<typename OtherDerived, int ... Idx>
  GroupProduct(const GroupProductBase<OtherDerived> & other, std::integer_sequence<int, Idx...>)
  : std::tuple<T<Scalar, Options>...>(std::get<Idx>(static_cast<const OtherDerived &>(other)) ...)
  {}

  template<int ... TBeg, int ... TLen>
  static GroupProduct<Scalar, Options, T...> exp_impl(
    const Tangent & v, std::integer_sequence<int, TBeg...>, std::integer_sequence<int, TLen...>
  )
  {
    return GroupProduct<Scalar, Options, T...>(
      T<Scalar, Options>::exp(v.template segment<TLen>(TBeg)) ...);
  }

  template<int ... TBeg, int ... TLen>
  static Tangent lieBracket_impl(
    const Tangent & a, const Tangent & b,
    std::integer_sequence<int, TBeg...>, std::integer_sequence<int, TLen...>
  )
  {
    Tangent res;
    ((res.template segment<TLen>(TBeg) = T<Scalar, Options>::lieBracket(
      a.template segment<TLen>(TBeg), b.template segment<TLen>(TBeg)
    )), ...);
    return res;
  }
};

}  // namespace cbr::lie


namespace Eigen
{

// Overload Eigen::Map<GroupProduct>
template<typename Scalar, int Options, template<typename, int> typename ... T>
class Map<cbr::lie::GroupProduct<Scalar, 0, T ...>, Options>
  : public cbr::lie::GroupProductBase<Map<cbr::lie::GroupProduct<Scalar, 0, T ...>, Options>>,
  public std::tuple<Eigen::Map<T<Scalar, 0>, Options>...>
{
  using Base = cbr::lie::GroupProductBase<Map<cbr::lie::GroupProduct<Scalar, 0, T ...>, Options>>;

public:
  using Base::operator=;
  using Base::operator*=;
  using Base::operator*;

  explicit Map(Scalar * x)
  : Map(x, typename Base::midx_beg_t{}) {}

private:
  template<int ... Beg>
  explicit Map(Scalar * x, std::integer_sequence<int, Beg ...>)
  : std::tuple<Eigen::Map<T<Scalar, 0>, Options>...>(Eigen::Map<T<Scalar, Options>>(x + Beg) ...)
  {}
};


// Overload Eigen::Map<const GroupProduct>
template<typename Scalar, int Options, template<typename, int> typename ... T>
class Map<const cbr::lie::GroupProduct<Scalar, 0, T ...>, Options>
  : public cbr::lie::GroupProductBase<Map<const cbr::lie::GroupProduct<Scalar, 0, T ...>,
    Options>>,
  public std::tuple<const Eigen::Map<const T<Scalar, 0>, Options>...>
{
  using Base =
    cbr::lie::GroupProductBase<Map<const cbr::lie::GroupProduct<Scalar, 0, T ...>, Options>>;

public:
  using Base::operator*;

  explicit Map(const Scalar * const x)
  : Map(x, typename Base::midx_beg_t{}) {}

private:
  template<int ... Beg>
  explicit Map(const Scalar * const x, std::integer_sequence<int, Beg ...>)
  : std::tuple<const Eigen::Map<const T<Scalar, 0>, Options>...>(Eigen::Map<const T<Scalar,
      Options>>(x + Beg) ...)
  {}
};


}  // namespace Eigen

#endif   // CBR_MATH__LIE__GROUP_PRODUCT_HPP_
