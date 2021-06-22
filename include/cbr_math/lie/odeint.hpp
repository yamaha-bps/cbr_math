// Copyright 2020 Yamaha Motor Corporation, USA

/*
Integration methods on Lie Groups

Changelog
=========
 * 2020-08: Created first version (Petter Nilsson)
 * 2020-09: Updated to work with standard lie group interface instead of product structure (Petter)

TODOs
=====
 * Implement cash_karp54 error stepper
 * Implement optimized schemes from [1] that uses fewer bracket evaluations

[0] Blanes, Sergio, and Fernando Casas. A Concise Introduction to Geometric Numerical Integration. Monographs and Research Notes in Mathematics. Chapman and Hall/CRC, 2016. https://doi.org/10.1201/b21563.
[1] Munthe–Kaas, Hans, and Brynjulf Owren. “Computations in a Free Lie Algebra.” Philosophical Transactions of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences 357, no. 1754 (April 15, 1999): 957–81. https://doi.org/10.1098/rsta.1999.0361.

*/


#ifndef BPS_LIBRARY__LIE__ODEINT_HPP_
#define BPS_LIBRARY__LIE__ODEINT_HPP_

#include <boost/numeric/odeint/stepper/stepper_categories.hpp>
#include <bps_library/utils.hpp>

#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include "common.hpp"

using std::get, std::numeric_limits;

namespace bps::lie::odeint
{

/**
 * Generic Runge-Kutta stepper for ODEs defined on Lie Groups.
 * @tparam Tableau butcher tableau with members A, B, C
 * @tparam State input space M1 x M2 x ... represented by std::tuple<M1, M2, ...>
 * @tparam Value data type for calculations
 * @tparam Deriv derivative space TM1 x TM2 x ... represented by std::tuple<TM1, TM2, ...>
 * @tparam Time data type for time calculations
 *
 * Implements the boost::numeric::odeint Stepper concept (basic stepper).
 *
 * A System
 *   (d/dt) xi  =  vi    xi \in Mi,  vi \in TMi
 * for i = 1, 2, ...
 *
 * is defined by a mapping std::tuple<M1, M2, ...> -> std::tuple<TM1, TM2, ...>
 * where TMi is the tangent space of Mi. Each Mi can be either a Sophus Lie group
 * or a regular Eigen member which is treated as the Lie group T(n) of translations.
 *
 * Example: A second-order system on SO(3) \times T SO(3) is given by the mapping
 * (q, \omega)  |-> (\omega, a)
 * where a is the angular acceleration (in R3).
 *
 * Remark: It is possible to optimize specific schemes since brackets sometimes
 * cancel and do not have to be calculated. This function focuses on being generic
 * and currently does not do such optimizations. See [1] at the top of the file.
 */
template<
  template<typename> typename Tableau,
  typename State,
  typename Value = double,
  typename Deriv = State,
  typename Time = Value
>
class runge_kutta
{
public:
  using state_type = State;
  using value_type = Value;
  using deriv_type = Deriv;
  using time_type = Time;
  using order_type = uint16_t;
  using stepper_category = boost::numeric::odeint::stepper_tag;

  static order_type order()
  {
    return Tableau<Value>::order;
  }

  template<class System>
  void do_step(System system, state_type & x, time_type t, time_type dt)
  {
    state_type xtmp;
    std::array<deriv_type, Tableau<Value>::R> u;
    std::array<deriv_type, Tableau<Value>::R> k;

    using R_t = std::decay_t<decltype(Tableau<Value>::R)>;

    // calculate ki's by iterating through butcher tableau
    bps::static_for<R_t, Tableau<Value>::R>(
      [&](auto s) {
        // calculate u[s] = \sum_j a[s][j] * dexpinv(u[j], k[j])
        u[s] = deriv_type::Zero();
        bps::static_for<R_t, s.value>(
          [&](auto j) {
            if (Tableau<Value>::A[s - 1][j] != 0) {
              u[s] += Tableau<Value>::A[s - 1][j] *
              dl_expinv_approx<state_type>(u[j], k[j], Tableau<Value>::order - 1);
            }
          });
        xtmp = x * state_type::exp(dt * u[s]);
        // calculate k[s] = f(X * exp(u[s] * dt), t + c[s] * dt)
        system(xtmp, k[s], t + Tableau<Value>::C[s] * dt);
      });

    // calculate step
    deriv_type ufin = deriv_type::Zero();
    bps::static_for_index<Tableau<Value>::R>(
      [&](auto j) {
        ufin += Tableau<Value>::B[j] * k[j];
      });
    x *= state_type::exp(dt * ufin);
  }
};

// Euler stepper as trivial rk scheme
template<typename V>
struct euler_tableau
{
  static constexpr uint16_t order = 1;
  static constexpr uint16_t R = 1;
  static constexpr std::array<std::array<V, R - 1>, R - 1> A{};
  static constexpr std::array<V, R> B{V(1)};
  static constexpr std::array<V, R> C{V(0)};
};
template<typename State, typename Value = double, typename Deriv = State, typename Time = Value>
using euler = runge_kutta<euler_tableau, State, Value, Deriv, Time>;

// Classic rk4 scheme
template<typename V>
struct rk4_tableau
{
  static constexpr uint16_t order = 4;
  static constexpr uint16_t R = 4;
  static constexpr std::array<std::array<V, R - 1>, R - 1> A{
    std::array<V, R - 1>{V(1) / 2, V(0), V(0)},
    std::array<V, R - 1>{V(0), V(1) / 2, V(0)},
    std::array<V, R - 1>{V(0), V(0), V(1)}
  };
  static constexpr std::array<V, R> B{V(1) / V(6), V(1) / V(3), V(1) / 3, V(1) / V(6)};
  static constexpr std::array<V, R> C{V(0), V(1) / V(2), V(1) / V(2), V(0)};
};
template<typename State, typename Value = double, typename Deriv = State, typename Time = Value>
using runge_kutta4 = runge_kutta<rk4_tableau, State, Value, Deriv, Time>;

// Not used for now, will be useful if/when implementing error steppers
template<typename V>
struct cash_karp_tableau
{
  static constexpr uint16_t order = 5;
  static constexpr uint16_t R = 6;
  static constexpr std::array<std::array<V, R - 1>, R - 1> A{
    std::array<V, R - 1>{V(1) / V(5), V(0), V(0), V(0), V(0)},
    std::array<V, R - 1>{V(3) / V(40), V(9) / V(40), V(0), V(0), V(0)},
    std::array<V, R - 1>{V(3) / V(10), V(-9) / V(10), V(6) / V(5), V(0), V(0)},
    std::array<V, R - 1>{-V(11) / V(54), V(5) / V(2), V(-70) / V(27), V(35) / V(27), V(0)},
    std::array<V, R - 1>{V(1631) / V(55296), V(175) / V(512), V(575) / V(13824),
      V(44275) / V(110592), V(253) / V(4096)}
  };
  static constexpr std::array<V, R> B{V(37) / V(378), V(0), V(250) / V(621), V(125) / V(594), V(0),
    V(512) / V(1771)};
  static constexpr std::array<V, R> C{V(0), V(1) / V(5), V(3) / V(10), V(3) / V(5), V(1),
    V(7) / V(8)};
};

}  // namespace bps::lie::odeint

#endif  // BPS_LIBRARY__LIE__ODEINT_HPP_
