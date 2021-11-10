// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_MATH__INTERP__PIECEWISE_POLY_HPP_
#define CBR_MATH__INTERP__PIECEWISE_POLY_HPP_

#include <Eigen/Dense>

#include <algorithm>
#include <functional>
#include <array>
#include <cstdint>
#include <exception>
#include <type_traits>
#include <utility>
#include <vector>
#include <tuple>

#include "cbr_math/math.hpp"
#include "cbr_math/eigen_traits.hpp"

// Forward declare YAML::convert
namespace YAML
{
template<typename T>
struct convert;
}  // namespace YAML

namespace cbr
{

namespace pp_details
{

class PiecewisePolyBase
{
protected:
  struct Extremities
  {
    Extremities() = default;
    Extremities(const Extremities &) = default;
    Extremities(Extremities &&) = default;
    Extremities & operator=(const Extremities &) = default;
    Extremities & operator=(Extremities &&) = default;
    ~Extremities() = default;

    template<typename T1, typename T2, typename T3, typename T4>
    Extremities(T1 && a, T2 && b, T3 && c, T4 && d)
    : val0{std::forward<T1>(a)},
      der0{std::forward<T2>(b)},
      valEnd{std::forward<T3>(c)},
      derEnd{std::forward<T4>(d)}
    {}

    double val0{};
    double der0{};
    double valEnd{};
    double derEnd{};
  };

public:
  enum class EXTRAP : uint16_t
  {
    CLAMP,
    LINEAR,
    YOLO
  };

  // Search method
  enum class SEARCH : uint16_t
  {
    LINEAR_FORWARD,
    LINEAR_REVERSE,
    LINEAR_HINT,
    BINARY
  };

public:
  template<typename T = double>
  using col_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  using row_t = Eigen::Matrix<double, 1, Eigen::Dynamic>;
  using matrix_t = Eigen::MatrixXd;

  PiecewisePolyBase() = delete;
  PiecewisePolyBase(const PiecewisePolyBase &) = default;
  PiecewisePolyBase(PiecewisePolyBase &&) = default;
  PiecewisePolyBase & operator=(const PiecewisePolyBase &) = default;
  PiecewisePolyBase & operator=(PiecewisePolyBase &&) = default;
  ~PiecewisePolyBase() = default;

  template<typename T1>
  PiecewisePolyBase(
    T1 && breakPoints,
    EXTRAP extrap,
    SEARCH search)
  : extrap_{extrap},
    search_{search},
    breakPoints_{std::forward<T1>(breakPoints)},
    length_{static_cast<std::size_t>(breakPoints_.cols())},
    x0_{breakPoints_.cols() > 0 ? breakPoints_[0] : 0.},
    xEnd_{breakPoints_.cols() > 0 ? breakPoints_[breakPoints_.cols() - 1] : 1.},
    hintSlope_{static_cast<double>(length_ - 1) / (xEnd_ - x0_)}
  {
    if (breakPoints_.cols() < 2) {
      throw std::invalid_argument(
              "Not enought break points to construct a PiecewisePoly, need at least 2.");
    }

    auto i2 = std::adjacent_find(
      breakPoints_.data(),
      breakPoints_.data() + breakPoints_.cols(),
      std::greater_equal<double>());

    if (i2 != (breakPoints_.data() + breakPoints_.cols())) {
      throw std::invalid_argument("PiecewisePoly's break points vector must be stricly sorted.");
    }
  }

  void set_search(const SEARCH search) noexcept
  {
    search_ = search;
  }

  void set_extrap(const EXTRAP extrap) noexcept
  {
    extrap_ = extrap;
  }

  const SEARCH & get_search() const noexcept
  {
    return search_;
  }

  const EXTRAP & get_extrap() const noexcept
  {
    return extrap_;
  }

  const row_t & breakPoints() const noexcept
  {
    return breakPoints_;
  }

  const double & lb() const noexcept
  {
    return x0_;
  }

  const double & ub() const noexcept
  {
    return xEnd_;
  }

  auto bounds() const noexcept
  {
    return std::pair<const double &, const double &>(lb(), ub());
  }

  const std::size_t & length() const noexcept
  {
    return length_;
  }

protected:
  std::pair<std::size_t, int8_t>
  findPosition(const double x) const
  {
    if (x < x0_) {
      return std::pair<std::size_t, int8_t>(0, -1);
    }

    if (x >= xEnd_) {
      return std::pair<std::size_t, int8_t>(0, 1);
    }

    std::size_t idx = 0;
    std::size_t flag = 0;

    if (length_ > 2) {
      switch (search_) {
        case SEARCH::LINEAR_FORWARD: {
            idx = 0;
            if (!findPositionLinear(true, x, idx)) {
              throw std::runtime_error("Couldn't find break point. Should never happen!");
            }
            break;
          }

        case SEARCH::LINEAR_REVERSE: {
            idx = length_ - 2;
            if (!findPositionLinear(false, x, idx)) {
              throw std::runtime_error("Couldn't find break point. Should never happen!");
            }
            break;
          }

        case SEARCH::LINEAR_HINT: {
            int64_t idxTmp = lround(hintSlope_ * (x - x0_));
            idx = static_cast<std::size_t>(idxTmp);
            if (idxTmp < 0 || idx > (length_ - 1)) {
              throw std::runtime_error("Point outside break points range. Should never happen!");
            }

            if (breakPoints_[static_cast<Eigen::Index>(idx)] >= x) {
              if (idx == length_ - 1) {
                idx--;
              }
              if (!findPositionLinear(false, x, idx)) {
                throw std::runtime_error("Couldn't find break point. Should never happen!");
              }
            } else {
              if (!findPositionLinear(true, x, idx)) {
                throw std::runtime_error("Couldn't find break point. Should never happen!");
              }
            }
            break;
          }

        case SEARCH::BINARY: {
            idx = 0;
            if (!findPositionBinary(x, idx)) {
              throw std::runtime_error("Couldn't find break point. Should never happen!");
            }
            break;
          }

        default: {
            throw std::runtime_error("Unkown search method. Should never happen!");
          }
      }
    }

    return std::pair<std::size_t, int8_t>(idx, flag);
  }

  /**
   * @brief Find index idx s.t. breakPts[idx] <= x < breakPts[idx+1] via linear search
   *
   * @param[in] forward search forward from starting index
   * @param[in] x
   * @param[in, out] idx starting index
   */
  bool findPositionLinear(
    const bool forward,
    const double x,
    std::size_t & idx) const noexcept
  {
    if (idx > (length_ - 2)) {
      return false;
    }

    if (forward) {
      for (; idx <= (length_ - 2); idx++) {
        if (breakPoints_[static_cast<Eigen::Index>(idx)] <= x &&
          x < breakPoints_[static_cast<Eigen::Index>(idx + 1)])
        {
          return true;
        }
      }
    } else {
      if (breakPoints_[0] <= x && x < breakPoints_[1]) {
        idx = 0;
        return true;
      }
      for (; idx > 0; idx--) {
        if (breakPoints_[static_cast<Eigen::Index>(idx)] <= x &&
          x < breakPoints_[static_cast<Eigen::Index>(idx + 1)])
        {
          return true;
        }
      }
    }

    return false;
  }

  /**
   * @brief Find index idx s.t. breakPts[idx] <= x < breakPts[idx+1] via binary search
   *
   * @param[in] x
   * @paran[out] idx
   */
  bool findPositionBinary(
    const double x,
    std::size_t & idx) const noexcept
  {
    if (length_ < 2 || x < breakPoints_[0] ||
      x >= breakPoints_[static_cast<Eigen::Index>(length_ - 1)])
    {
      return false;
    }

    std::size_t low{0}, high{length_ - 1};

    while (low < high) {
      const double frac = (x - breakPoints_[static_cast<Eigen::Index>(low)]) /
        (breakPoints_[static_cast<Eigen::Index>(high)] -
        breakPoints_[static_cast<Eigen::Index>(low)]);
      idx = low + static_cast<std::size_t>(frac * static_cast<double>(high - low));
      if (x < breakPoints_[static_cast<Eigen::Index>(idx)]) {
        high = idx;
      } else if (x >= breakPoints_[static_cast<Eigen::Index>(idx) + 1]) {
        low = idx + 1;
      } else {
        break;  // we have breakPoints_[idx] <= t < breakPoints_[idx+1]
      }
    }

    return true;
  }

  template<typename T>
  T evalPoly(
    const T x,
    const std::size_t pieceIdx,
    const matrix_t & coefList,
    const std::size_t order) const
  {
    T val =
      static_cast<T>(coefList(
        static_cast<Eigen::Index>(order - 1),
        static_cast<Eigen::Index>(pieceIdx)));
    const auto & x0 = breakPoints_[static_cast<Eigen::Index>(pieceIdx)];
    for (std::size_t i = 1; i < order; i++) {
      val +=
        static_cast<T>(coefList(
          static_cast<Eigen::Index>(order - 1 - i),
          static_cast<Eigen::Index>(pieceIdx))) * powFast<T, std::size_t>(
        x - static_cast<T>(x0), i);
    }
    return val;
  }

  template<typename T>
  T evalPolyDer(
    const T x,
    const std::size_t pieceIdx,
    const std::size_t n,
    const matrix_t & coefList,
    const std::size_t order) const
  {
    if (n == 0) {
      return evalPoly(x, pieceIdx, coefList, order);

    } else {
      T val{0.};

      if (n < order) {
        const auto & x0 = breakPoints_[static_cast<Eigen::Index>(pieceIdx)];
        for (std::size_t i = n; i < order; i++) {
          val +=
            static_cast<T>(factorial(i, i - n + 1)) * static_cast<T>(coefList(
              static_cast<Eigen::Index>(order - 1 - i),
              static_cast<Eigen::Index>(pieceIdx))) *
            powFast<T, std::size_t>(
            x - static_cast<T>(x0),
            i - n);
        }
      }

      return val;
    }
  }

  template<typename T>
  T extrapPoly(
    const int8_t direction,
    const T x,
    const std::size_t n,
    const matrix_t & coefList,
    const std::size_t order,
    const Extremities & ex) const
  {
    if (direction == 1) {
      switch (extrap_) {
        case EXTRAP::CLAMP: {
            if (n == 0) {
              return static_cast<T>(ex.valEnd);
            }
            return T{0.};
          }

        case EXTRAP::LINEAR: {
            if (n == 0) {
              return static_cast<T>(ex.valEnd + ex.derEnd * (x - xEnd_));
            }
            if (n == 1) {
              return static_cast<T>(ex.derEnd);
            }
            return T{0.};
          }

        case EXTRAP::YOLO: {
            if (n == 0) {
              return evalPoly(x, length_ - 2, coefList, order);
            }
            return evalPolyDer(x, length_ - 2, n, coefList, order);
          }

        default: {
            throw std::runtime_error("Unkown extrapolation method. Should never happen!");
          }
      }
    } else if (direction == -1) {
      switch (extrap_) {
        case EXTRAP::CLAMP: {
            if (n == 0) {
              return static_cast<T>(ex.val0);
            }
            return T{0.};
          }
        case EXTRAP::LINEAR: {
            if (n == 0) {
              return static_cast<T>(ex.val0 + ex.der0 * (x - x0_));
            }
            if (n == 1) {
              return static_cast<T>(ex.der0);
            }
            return T{0.};
          }

        case EXTRAP::YOLO: {
            if (n == 0) {
              return evalPoly(x, 0, coefList, order);
            }
            return evalPolyDer(x, 0, n, coefList, order);
          }

        default: {
            throw std::runtime_error("Unkown extrapolation method. Should never happen!");
          }
      }
    } else {
      throw std::runtime_error("Invalid extrapolation direction. Should never happen!");
    }
  }

protected:
  EXTRAP extrap_{};
  SEARCH search_{};
  row_t breakPoints_{};
  std::size_t length_{};
  double x0_{};
  double xEnd_{};
  double hintSlope_{};
};

}  // namespace pp_details

/***************************************************************************
 * \brief Class for a scalar piecewise polynomial with scalar image
 ***************************************************************************/
class PiecewisePoly : public pp_details::PiecewisePolyBase
{
public:
  PiecewisePoly() = delete;
  PiecewisePoly(const PiecewisePoly &) = default;
  PiecewisePoly(PiecewisePoly &&) = default;
  PiecewisePoly & operator=(const PiecewisePoly &) = default;
  PiecewisePoly & operator=(PiecewisePoly &&) = default;
  ~PiecewisePoly() = default;


  /**
   * @brief Scalar-valued piecewise polynomial
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 PiecewisePoly::matrix_t
   * @param breakPoints range breakpoints of shape 1xN
   * @param coefList polynomial coefficients of shape DxN
   * @param extrap extrapolation method [CLAMP/LINEAR/YOLO]
   * @param search search method [LINEAR_FORWARD/LINEAR_REVERSE/LINEAR_HINT/BINARY]
   *
   * The polynomial is s.t.
   *
   * p(x) = p_i(t - breakPoints[i])   if breakPoints[i] <= t < breakPoints[i+1]
   *
   *   where p_i is the polynimial
   *
   * p_i(t) = coefList[0, i] + coefList[1, i] * t + ... + coefList[D-1, i] * t^{D-1}
   *
   * For values outside of the range [breakPoints[0], breakPoints[breakPoints.size() - 1]]
   * the value is extrapolated using the given method.
   */
  template<typename T1, typename T2>
  PiecewisePoly(
    T1 && breakPoints,
    T2 && coefList,
    EXTRAP extrap = EXTRAP::CLAMP,
    SEARCH search = SEARCH::LINEAR_HINT)
  : pp_details::PiecewisePolyBase(
      std::forward<T1>(breakPoints),
      extrap,
      search),
    coefList_{std::forward<T2>(coefList)},
    order_{static_cast<std::size_t>(coefList_.rows())}
  {
    if (breakPoints_.cols() != (coefList_.cols() + 1)) {
      throw std::invalid_argument(
              "PiecewisePoly coefficient matrix must have "
              "one less column that the number of break points.");
    }

    if (coefList_.rows() < 1) {
      throw std::invalid_argument("PiecewisePoly coefficient matrix must have a least 1 row.");
    }

    extremities_.val0 = evalPoly<double>(breakPoints_[0], 0, coefList_, order_);
    extremities_.der0 = evalPolyDer<double>(breakPoints_[0], 0, 1, coefList_, order_);
    extremities_.valEnd = evalPoly<double>(
      breakPoints_[static_cast<Eigen::Index>(length_ - 1)], length_ - 2, coefList_, order_);
    extremities_.derEnd = evalPolyDer<double>(
      breakPoints_[static_cast<Eigen::Index>(length_ - 1)], length_ - 2, 1, coefList_,
      order_);
  }

  template<typename T = double>
  T val(const T x) const
  {
    const auto [pieceIdx, flag] = findPosition(static_cast<double>(x));
    if (flag == 0) {
      return evalPoly<T>(x, pieceIdx, coefList_, order_);
    }
    return extrapPoly<T>(flag, x, 0, coefList_, order_, extremities_);
  }

  template<typename Derived>
  auto vals(const Eigen::MatrixBase<Derived> & xs) const
  {
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime,
      Derived::ColsAtCompileTime> out(xs.rows(), xs.cols());
    for (Eigen::Index i = 0; i < xs.rows(); i++) {
      for (Eigen::Index j = 0; j < xs.cols(); j++) {
        out(i, j) = val<typename Derived::Scalar>(xs(i, j));
      }
    }
    return out;
  }

  template<typename T, std::size_t N>
  auto vals(const std::array<T, N> & xs) const
  {
    static_assert(N > 0, "Input must be of size > 0.");
    std::array<T, N> out;
    for (size_t i = 0; i < N; i++) {
      out[i] = val<T>(xs[i]);
    }
    return out;
  }

  template<typename T>
  auto vals(const std::vector<T> & xs) const
  {
    std::vector<T> out;
    out.reserve(xs.size());
    for (const auto & x : xs) {
      out.push_back(val<T>(x));
    }
    return out;
  }

  template<typename T = double>
  T der(const T x, const std::size_t n = 1) const
  {
    const auto [pieceIdx, flag] = findPosition(static_cast<double>(x));
    if (flag == 0) {
      return evalPolyDer<T>(x, pieceIdx, n, coefList_, order_);
    }
    return extrapPoly<T>(flag, x, n, coefList_, order_, extremities_);
  }

  template<typename Derived>
  auto ders(const Eigen::MatrixBase<Derived> & xs, const std::size_t n = 1) const
  {
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime,
      Derived::ColsAtCompileTime> out(xs.rows(), xs.cols());
    for (Eigen::Index i = 0; i < xs.rows(); i++) {
      for (Eigen::Index j = 0; j < xs.cols(); j++) {
        out(i, j) = der<typename Derived::Scalar>(xs(i, j), n);
      }
    }
    return out;
  }

  template<typename T, std::size_t N>
  auto ders(const std::array<T, N> & xs, const std::size_t n = 1) const
  {
    static_assert(N > 0, "Input must be of size > 0.");

    std::array<T, N> out;
    for (size_t i = 0; i < N; i++) {
      out[i] = der<T>(xs[i], n);
    }
    return out;
  }

  template<typename T>
  auto ders(const std::vector<T> & xs, const std::size_t n = 1) const
  {
    std::vector<T> out;
    out.reserve(xs.size());
    for (const auto & x : xs) {
      out.push_back(der<T>(x, n));
    }
    return out;
  }

  const matrix_t & coeffs() const noexcept
  {
    return coefList_;
  }

  const std::size_t & order() const noexcept
  {
    return order_;
  }

protected:
  matrix_t coefList_{};
  std::size_t order_{};
  Extremities extremities_{};

  template<typename T>
  friend class YAML::convert;
};

/**************************************************************************
 * \brief Class for a scalar piecewise polynomial with multivalued image
 **************************************************************************/
class PiecewisePolyND : public pp_details::PiecewisePolyBase
{
public:
  PiecewisePolyND() = delete;
  PiecewisePolyND(const PiecewisePolyND &) = default;
  PiecewisePolyND(PiecewisePolyND &&) = default;
  PiecewisePolyND & operator=(const PiecewisePolyND &) = default;
  PiecewisePolyND & operator=(PiecewisePolyND &&) = default;
  ~PiecewisePolyND() = default;

  /**
   * @brief Vector-valued piecewise polynomial
   *
   * @tparam T1 PiecewisePoly::row_t
   * @tparam T2 container_t<PiecewisePoly::matrix_t>
   * @param breakPoints range breakpoints of shape 1xN
   * @param coefLists polynomial coefficients of shape MxDxN
   * @param extrap extrapolation method [CLAMP/LINEAR/YOLO]
   * @param search search method [LINEAR_FORWARD/LINEAR_REVERSE/LINEAR_HINT/BINARY]
   *
   * The polynomial is s.t.
   *
   * p(x) = p_i(t - breakPoints[i])   if breakPoints[i] <= t < breakPoints[i+1]
   *
   *   where p_i is the polynimial
   *
   * p_i[m](t) = coefList[m, 0, i] + coefList[m, 1, i] * t + ... + coefList[m, D-1, i] * t^{D-1}
   *
   * For values outside of the range [breakPoints[0], breakPoints[breakPoints.size() - 1]]
   * the value is extrapolated using the given method.
   */
  template<typename T1, typename T2>
  PiecewisePolyND(
    T1 && breakPoints,
    T2 && coefLists,
    EXTRAP extrap = EXTRAP::CLAMP,
    SEARCH search = SEARCH::LINEAR_HINT)
  : pp_details::PiecewisePolyBase(
      std::forward<T1>(breakPoints),
      extrap,
      search),
    coefLists_{std::forward<T2>(coefLists)},
    N_{coefLists_.size()}
  {
    if (N_ < 1) {
      throw std::invalid_argument("coefList size must be > 0.");
    }

    orders_.reserve(N_);
    extremities_.reserve(N_);

    for (const auto & coefList : coefLists_) {
      if (breakPoints_.cols() != (coefList.cols() + 1)) {
        throw std::invalid_argument(
                "PiecewisePoly coefficient matrix must have "
                "one less column that the number of break points.");
      }

      if (coefList.rows() < 1) {
        throw std::invalid_argument("PiecewisePoly coefficient matrix must have a least 1 row.");
      }

      orders_.push_back(static_cast<std::size_t>(coefList.rows()));

      extremities_.emplace_back(
        evalPoly<double>(breakPoints_[0], 0, coefList, orders_.back()),
        evalPolyDer<double>(breakPoints_[0], 0, 1, coefList, orders_.back()),
        evalPoly<double>(
          breakPoints_[static_cast<Eigen::Index>(length_ - 1)], length_ - 2,
          coefList, orders_.back()),
        evalPolyDer<double>(
          breakPoints_[static_cast<Eigen::Index>(length_ - 1)], length_ - 2, 1,
          coefList, orders_.back())
      );
    }
  }

  template<typename T = double>
  col_t<T> val(const T x) const
  {
    col_t<T> out(N_);

    const auto [pieceIdx, flag] = findPosition(static_cast<double>(x));
    if (flag == 0) {
      for (std::size_t i = 0; i < N_; i++) {
        out[static_cast<Eigen::Index>(i)] = evalPoly<T>(x, pieceIdx, coefLists_[i], orders_[i]);
      }
    } else {
      for (std::size_t i = 0; i < N_; i++) {
        out[static_cast<Eigen::Index>(i)] = extrapPoly<T>(
          flag, x, 0, coefLists_[i], orders_[i],
          extremities_[i]);
      }
    }

    return out;
  }

  template<typename Derived>
  auto vals(const Eigen::MatrixBase<Derived> & xs) const
  {
    static_assert(Derived::IsVectorAtCompileTime, "Input must be a vector.");

    Eigen::Matrix<typename Derived::Scalar,
      Eigen::Dynamic, Derived::SizeAtCompileTime> out(N_, xs.size());

    for (Eigen::Index i = 0; i < xs.size(); i++) {
      out.col(i) = val<typename Derived::Scalar>(xs[i]);
    }

    return out;
  }

  template<typename T, std::size_t N>
  auto vals(const std::array<T, N> & xs) const
  {
    static_assert(N > 0, "Input must be of size > 0.");
    std::array<col_t<T>, N> out;
    for (size_t i = 0; i < N; i++) {
      out[i] = val<T>(xs[i]);
    }
    return out;
  }

  template<typename T>
  auto vals(const std::vector<T> & xs) const
  {
    std::vector<col_t<T>> out;
    out.reserve(xs.size());
    for (const auto & x : xs) {
      out.push_back(val<T>(x));
    }
    return out;
  }

  template<typename T = double>
  col_t<T> der(const T x, const std::size_t n = 1) const
  {
    col_t<T> out(N_);

    const auto [pieceIdx, flag] = findPosition(static_cast<double>(x));
    if (flag == 0) {
      for (std::size_t i = 0; i < N_; i++) {
        out[static_cast<Eigen::Index>(i)] =
          evalPolyDer<T>(x, pieceIdx, n, coefLists_[i], orders_[i]);
      }
    } else {
      for (std::size_t i = 0; i < N_; i++) {
        out[static_cast<Eigen::Index>(i)] = extrapPoly<T>(
          flag, x, n, coefLists_[i], orders_[i],
          extremities_[i]);
      }
    }

    return out;
  }

  template<typename Derived>
  auto ders(const Eigen::MatrixBase<Derived> & xs, const std::size_t n = 1) const
  {
    static_assert(Derived::IsVectorAtCompileTime, "Input must be a vector.");

    Eigen::Matrix<typename Derived::Scalar,
      Eigen::Dynamic, Derived::SizeAtCompileTime> out(N_, xs.size());

    for (Eigen::Index i = 0; i < xs.size(); i++) {
      out.col(i) = der<typename Derived::Scalar>(xs[i], n);
    }

    return out;
  }

  template<typename T, std::size_t N>
  auto ders(const std::array<T, N> & xs, const std::size_t n = 1) const
  {
    static_assert(N > 0, "Input must be of size > 0.");

    std::array<col_t<T>, N> out;
    for (std::size_t i = 0; i < N; i++) {
      out[i] = der<T>(xs[i], n);
    }
    return out;
  }

  template<typename T>
  auto ders(const std::vector<T> & xs, const std::size_t n = 1) const
  {
    std::vector<col_t<T>> out;
    out.reserve(xs.size());
    for (const auto & x : xs) {
      out.push_back(der<T>(x, n));
    }
    return out;
  }

  const std::size_t & dimension() const noexcept
  {
    return N_;
  }

  const matrix_t & coeffs(const std::size_t i) const
  {
    if (i >= N_) {
      throw std::out_of_range(
              "coeffs was call with an index greater than the dimension of the polynomial.");
    }
    return coefLists_[i];
  }

  const std::size_t & order(const std::size_t i) const
  {
    if (i <= N_) {
      throw std::out_of_range(
              "getCoeffs was call with an index greater than the dimension of the polynomial.");
    }
    return orders_[i];
  }

protected:
  std::vector<matrix_t> coefLists_;
  std::size_t N_ = 0;
  std::vector<std::size_t> orders_;
  std::vector<Extremities> extremities_;
};

}  // namespace cbr

#endif  // CBR_MATH__INTERP__PIECEWISE_POLY_HPP_
