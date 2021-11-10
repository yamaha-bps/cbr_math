// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_MATH__MATH_HPP_
#define CBR_MATH__MATH_HPP_

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <limits>
#include <type_traits>

namespace cbr
{

/***************************************************************************
 * \brief Converts degrees to radians
 ***************************************************************************/
template<typename T>
constexpr T deg2rad(const T deg) noexcept
{
  static_assert(std::is_floating_point_v<T>, "Input type must be a floating point.");
  return deg * M_PI / 180.;
}

/***************************************************************************
 * \brief Converts radians to degrees
 ***************************************************************************/
template<typename T>
constexpr T rad2deg(const T rad) noexcept
{
  static_assert(std::is_floating_point_v<T>, "Input type must be a floating point.");
  return rad * 180. / M_PI;
}

/***************************************************************************
 * \brief Computes b^e
 ***************************************************************************/
template<typename TBase, typename TExp>
constexpr TBase powFast(const TBase b, const TExp e) noexcept
{
  static_assert(std::is_unsigned_v<TExp>, "Exponent type must be unsigned.");

  if (e == 0) {
    return TBase(1.);
  }

  if (e == 1) {
    return b;
  }

  TBase out = b;
  for (TExp i = 2; i <= e; i++) {
    out *= b;
  }

  return out;
}

namespace detail
{

template<typename T, std::size_t ... Is>
constexpr T powFastImpl([[maybe_unused]] T val, std::index_sequence<Is...>)
{
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-value"
  #pragma GCC diagnostic ignored "-Wconversion"
  // *INDENT-OFF*
  return ((Is, val) * ... * T{1});
  // *INDENT-ON*
  #pragma GCC diagnostic pop
}

}  // namespace detail

// /***************************************************************************
//  * \brief Computes b^e via expansion to e-length product  b * ... * b
//  ***************************************************************************/
template<std::size_t e = 2, typename TBase>
constexpr TBase powFast(const TBase b) noexcept
{
  return detail::powFastImpl(b, std::make_index_sequence<e>{});
}

/***************************************************************************
* \brief Wraps angle in radians between 0 and 2*pi
***************************************************************************/
template<typename T>
T wrap2Pi(const T in) noexcept
{
  static_assert(std::is_floating_point_v<T>, "Input type must be a floating point.");
  return in - std::floor(in / T(2. * M_PI)) * T(2. * M_PI);
}

/***************************************************************************
* \brief Wraps angle in radians between -pi and pi
***************************************************************************/
template<typename T>
T wrapPi(const T in) noexcept
{
  static_assert(std::is_floating_point_v<T>, "Input type must be a floating point.");
  return wrap2Pi(in + M_PI) - M_PI;
}

/***************************************************************************
 * \brief Converts yaw in radians to heading in degrees
 ***************************************************************************/
template<typename T>
T yaw2heading(const T rad) noexcept
{
  static_assert(std::is_floating_point_v<T>, "Input type must be a floating point.");
  return rad2deg(wrap2Pi(-rad));
}

/***************************************************************************
 * \brief Computes truncated factorial : product from x0(default: 1) to x.
 ***************************************************************************/
template<typename T>
constexpr T factorial(const T x, const T x0 = T(1)) noexcept
{
  static_assert(std::is_unsigned_v<T>, "Type must be unsigned.");

  if (x == 0 || x0 > x) {
    return T(1);
  }

  if (x0 >= x) {
    return x;
  }

  T out = x0;
  for (T i = x0 + T(1); i <= x; i++) {
    out *= i;
  }
  return out;
}


/***************************************************************************
 * \brief Converts probability to log odds
 ***************************************************************************/
template<typename T>
constexpr T prob_to_log_odds(const T p)
{
  static_assert(std::is_floating_point_v<T>, "Input type must be a floating point.");
  return log(p / (1. - p));
}

/***************************************************************************
 * \brief Converts to log odds to probability
 ***************************************************************************/
template<typename T>
constexpr T log_odds_to_prob(const T l)
{
  static_assert(std::is_floating_point_v<T>, "Input type must be a floating point.");
  return 1. - (1. / (1. + exp(l)));
}

/***************************************************************************
 * \brief Converts a quaternion to euler angles using the ZYX decomposition
 * The returned vector contains the angles in [roll, pitch, yaw] order
 ***************************************************************************/
template<typename derived>
Eigen::Matrix<typename derived::Scalar, 3, 1>
quat2eulZYX(const Eigen::QuaternionBase<derived> & q) noexcept
{
  const auto qn = q.normalized();
  using T = typename derived::Scalar;
  Eigen::Matrix<T, 3, 1> eul;

  // roll
  eul[0] = atan2(
    T(2.) * (qn.w() * qn.x() + qn.y() * qn.z()),
    T(1.) - T(2.) * (powFast<2>(qn.x()) + powFast<2>(qn.y())));

  // pitch
  const T sinp = T(2.) * (qn.w() * qn.y() - qn.z() * qn.x());
  // Autodiff friendly manner of handling numerical errors
  if (abs(sinp) >= T(1.)) {
    if (sinp > T(0.)) {
      eul[1] = T(M_PI_2);
    } else {
      eul[1] = -T(M_PI_2);
    }
  } else {
    eul[1] = asin(sinp);
  }

  // yaw
  eul[2] = atan2(
    T(2.) * (qn.w() * qn.z() + qn.x() * qn.y()),
    T(1.) - T(2.) * (powFast<2>(qn.y()) + powFast<2>(qn.z())));

  return eul;
}


/***************************************************************************
 * \brief Converts a quaternion to euler angles using the ZYX decomposition
 * The returned vector contains the angles in [roll, pitch, yaw] order
 ***************************************************************************/
template<typename derived>
void quat2eulZYX(
  const Eigen::QuaternionBase<derived> & q,
  Eigen::Matrix<typename derived::Scalar, 3, 1> & eul) noexcept
{
  eul = quat2eulZYX<derived>(q);
}

/***************************************************************************
 * \brief Converts euler angles to a quaternion using the ZYX decomposition
 * The input vector must contain the angles in [roll, pitch, yaw] order
 ***************************************************************************/
template<typename T>
Eigen::Quaternion<T> eul2quatZYX(const Eigen::Matrix<T, 3, 1> & eul) noexcept
{
  Eigen::Quaternion<T> q;

  T cy = cos(eul[2] * .5);
  T sy = sin(eul[2] * .5);
  T cp = cos(eul[1] * .5);
  T sp = sin(eul[1] * .5);
  T cr = cos(eul[0] * .5);
  T sr = sin(eul[0] * .5);

  q.w() = cy * cp * cr + sy * sp * sr;
  q.x() = cy * cp * sr - sy * sp * cr;
  q.y() = sy * cp * sr + cy * sp * cr;
  q.z() = sy * cp * cr - cy * sp * sr;

  return q;
}

template<typename T, typename derived>
void eul2quatZYX(
  const Eigen::Matrix<T, 3, 1> & eul,
  Eigen::QuaternionBase<derived> & q) noexcept
{
  q = eul2quatZYX<T>(eul);
}

/***************************************************************************
 * \brief Converts a pure yaw to a quaternion
 ***************************************************************************/
template<typename T>
Eigen::Quaternion<T> yaw2quat(const T & yaw) noexcept
{
  return Eigen::Quaternion<T>(cos(yaw * T(.5)), T(0.0), T(0.0), sin(yaw * T(.5)));
}

/***************************************************************************
 * \brief Extract the yaw component of a quaternion
 ***************************************************************************/
template<typename derived>
typename derived::Scalar quat2yaw(const Eigen::QuaternionBase<derived> & q) noexcept
{
  using T = typename derived::Scalar;

  const auto qn = q.normalized();
  return atan2(
    T(2.) * (qn.w() * qn.z() + qn.x() * qn.y()),
    T(1.) - T(2.) * (powFast<2>(qn.y()) + powFast<2>(qn.z())));
}


/***************************************************************************
 * \brief Convert subscripts to linear indices
 ***************************************************************************/
template<typename T, std::size_t N>
constexpr T sub2ind(const std::array<T, N> & sz, const std::array<T, N> & idx)
{
  static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic.");
  static_assert(N > 1, "Size of inputs must be > 1.");

  T ind = idx[0] + idx[1] * sz[0];

  if constexpr (N > 2) {
    T stride = sz[0];
    for (std::size_t i = 2; i < N; i++) {
      stride *= sz[i - 1];
      ind += idx[i] * stride;
    }
  }

  return ind;
}

/***************************************************************************
 * \brief Convert linear indices to subscripts
 ***************************************************************************/
template<typename T, std::size_t N>
constexpr std::array<T, N> ind2sub(const std::array<T, N> & sz, T idx)
{
  static_assert(std::is_arithmetic_v<T>, "Type must be arithmetic.");
  static_assert(N > 1, "Size of first input must be > 1.");

  std::array<T, N> sub{};

  if constexpr (N > 2) {
    std::array<T, N - 2> strides{};
    strides[0] = sz[0] * sz[1];
    for (std::size_t i = 1; i < N - 2; i++) {
      strides[i] = strides[i - 1] * sz[i + 1];
    }
    for (std::size_t i = N - 1; i > 1; i--) {
      sub[i] = idx / strides[i - 2];
      idx = idx % strides[i - 2];
    }
  }

  sub[1] = idx / sz[0];
  sub[0] = idx % sz[0];

  return sub;
}

/***************************************************************************
 * \brief Antisymmetric power function
 ***************************************************************************/
template<typename T1, typename T2>
auto powAntisym(T1 base, T2 exp)
{
  static_assert(std::is_arithmetic_v<T1>, "Base type must be arithmetic.");
  static_assert(std::is_arithmetic_v<T2>, "Exponent type must be arithmetic.");

  if (base >= T1(0)) {
    return std::pow(base, exp);
  } else {
    return -std::pow(-base, exp);
  }
}


/***************************************************************************
 * \brief Over-approximate a body bounding box with a world bounding box
 ***************************************************************************/
template<typename S, typename T, int I>
Eigen::AlignedBox<T, I> overapp_bbox(S && pose, Eigen::AlignedBox<T, I> bbox_B)
{
  Eigen::Array<T, I, 1> bbmin = std::numeric_limits<T>::max() * Eigen::Array<T, I, 1>::Ones();
  Eigen::Array<T, I, 1> bbmax = std::numeric_limits<T>::min() * Eigen::Array<T, I, 1>::Ones();
  for (int n = 0; n != pow(2, I); ++n) {  // find min/max coordinates of bounding box in world frame
    auto p = pose * bbox_B.corner(static_cast<typename Eigen::AlignedBox<T, I>::CornerType>(n));
    bbmin = bbmin.min(p.array());
    bbmax = bbmax.max(p.array());
  }
  return Eigen::AlignedBox<T, I>{bbmin, bbmax};
}

/***************************************************************************
 * \brief Smooth saturation function
 ***************************************************************************/
template<typename T1, typename T2>
constexpr void smoothSatInPlace(
  T1 & x,
  const T2 & mi,
  const T2 & ma,
  const T2 & satSmoothCoeff = T2(0.1))
{
  static_assert(std::is_floating_point_v<T2>, "Input type must be a floating point.");

  if (satSmoothCoeff < T2(0.) || T2(1.) < satSmoothCoeff) {
    throw std::invalid_argument("satSmoothCoeff must be between 0 and 1.");
  }
  if (ma < mi) {
    throw std::invalid_argument("max must be greated or equal to min.");
  }

  constexpr T2 alpha = M_PI / 8.;
  constexpr T2 beta = M_PI / 4.;
  const T2 r = satSmoothCoeff * M_SQRT2 / tan(alpha);
  const T2 bevelL = r * tan(alpha);
  const T2 bevelStart = 1. - cos(beta) * bevelL;
  const T2 bevelStop = 1. + bevelL;
  const T2 bevelXc = bevelStop;
  const T2 bevelYc = 1. - r;

  const T2 range = ma - mi;
  const T2 middle = (ma + mi) / 2.;
  const T1 uc = 2. * (x - middle) / range;

  if (uc >= bevelStop) {
    x = ma;
  } else if (uc <= -bevelStop) {
    x = mi;
  } else if (uc > bevelStart) {
    x = 0.5 * (sqrt(r * r - (uc - bevelXc) * (uc - bevelXc)) + bevelYc) * range + middle;
  } else if (uc < -bevelStart) {
    x = 0.5 * (-sqrt(r * r - (uc + bevelXc) * (uc + bevelXc)) - bevelYc) * range + middle;
  } else {
    return;
  }
}

template<typename T1, typename T2>
constexpr T1 smoothSat(
  const T1 & x,
  const T2 & mi,
  const T2 & ma,
  const T2 & satSmoothCoeff = T2{0.1})
{
  T1 out{x};
  smoothSatInPlace<T1, T2>(out, mi, ma, satSmoothCoeff);
  return out;
}

/***************************************************************************
 * \brief Check if point is inside polygon
 ***************************************************************************/
template<typename ForwardIterator, typename Point>
bool point_in_polygon(
  ForwardIterator first,
  ForwardIterator last,
  const Point & point)
{
  ForwardIterator current = first;
  if (current == last) {return false;}  // empty list of vertices

  ForwardIterator next = current; ++next;
  if (next == last) {return false;}  // only 1 vertex

  ForwardIterator next_plus_1 = next; ++next_plus_1;
  if (next_plus_1 == last) {return false;}  // only 2 vertex

  auto which_side_in_slab =
    [](
    const auto & pt,
    const auto & low,
    const auto & high) -> int
    {
      const double cross_product =
        (high[0] - low[0]) * (pt[1] - low[1]) - (high[1] - low[1]) * (pt[0] - low[0]);

      if (cross_product > 0.) {
        return 1;
      }

      if (cross_product < 0.) {
        return -1;
      }

      return 0;
    };

  auto compare_x_2 = [](const auto & p1, const auto & p2) -> int {
      if (p1[0] < p2[0]) {
        return -1;
      } else if (p1[0] > p2[0]) {
        return 1;
      }
      return 0;
    };

  auto compare_y_2 = [](const auto & p1, const auto & p2) -> int {
      if (p1[1] < p2[1]) {
        return -1;
      } else if (p1[1] > p2[1]) {
        return 1;
      }
      return 0;
    };

  bool IsInside = false;
  int cur_y_comp_res = compare_y_2(*current, point);

  do {
    int next_y_comp_res = compare_y_2(*next, point);

    switch (cur_y_comp_res) {
      case -1:
        switch (next_y_comp_res) {
          case -1:
            break;
          case 0:
            switch (compare_x_2(point, *next)) {
              case -1:
                {IsInside = !IsInside; break;}
              case 0:
                return true;
              case 1:
                break;
            }
            break;
          case 1:
            switch (which_side_in_slab(point, *current, *next)) {
              case -1:
                {IsInside = !IsInside; break;}
              case 0:
                return true;
            }
            break;
        }
        break;
      case 0:
        switch (next_y_comp_res) {
          case -1:
            switch (compare_x_2(point, *current)) {
              case -1:
                {IsInside = !IsInside; break;}
              case 0:
                return true;
              case 1:
                break;
            }
            break;
          case 0:
            switch (compare_x_2(point, *current)) {
              case -1:
                if (compare_x_2(point, *next) != -1) {
                  return true;
                }
                break;
              case 0:
                return true;
              case 1:
                if (compare_x_2(point, *next) != 1) {
                  return true;
                }
                break;
            }
            break;
          case 1:
            if (compare_x_2(point, *current) == 0) {
              return true;
            }
            break;
        }
        break;
      case 1:
        switch (next_y_comp_res) {
          case -1:
            switch (which_side_in_slab(point, *next, *current)) {
              case -1:
                {IsInside = !IsInside; break;}
              case 0:
                return true;
            }
            break;
          case 0:
            if (compare_x_2(point, *next) == 0) {
              return true;
            }
            break;
          case 1:
            break;
        }
        break;
    }
    current = next;
    cur_y_comp_res = next_y_comp_res;
    ++next;
    if (next == last) {next = first;}
  } while (current != first);

  return IsInside;
}

/***************************************************************************
 * \brief Compute distance from point to segment
 ***************************************************************************/
template<typename Point_t>
constexpr auto point_to_segment_dist_squared(
  const Point_t & pt,
  const Point_t & seg_pt1,
  const Point_t & seg_pt2)
{
  using scalar_t = std::decay_t<decltype(pt[0])>;

  const scalar_t pt_x = pt[0] - seg_pt1[0];
  const scalar_t pt_y = pt[1] - seg_pt1[1];

  if (seg_pt1[0] == seg_pt2[0] && seg_pt1[1] == seg_pt2[1]) {
    return pt_x * pt_x + pt_y * pt_y;
  }

  const scalar_t seg_x = seg_pt2[0] - seg_pt1[0];
  const scalar_t seg_y = seg_pt2[1] - seg_pt1[1];

  const scalar_t l2 = seg_x * seg_x + seg_y * seg_y;
  const scalar_t proj = (pt_x * seg_x + pt_y * seg_y) / l2;
  const scalar_t t = std::min(std::max(proj, scalar_t(0.)), scalar_t(1.));

  const scalar_t dist_x = t * seg_x - pt_x;
  const scalar_t dist_y = t * seg_y - pt_y;

  return dist_x * dist_x + dist_y * dist_y;
}


/***************************************************************************
 * \brief Checks if 2 segments intersect
 ***************************************************************************/
template<typename Point_t>
constexpr bool segments_intersect(
  const Point_t & seg1_pt1,
  const Point_t & seg1_pt2,
  const Point_t & seg2_pt1,
  const Point_t & seg2_pt2)
{
  if (seg1_pt1[0] == seg1_pt2[0] && seg1_pt1[1] == seg1_pt2[1]) {
    throw std::invalid_argument("First segment has no length");
  }

  if (seg2_pt1[0] == seg2_pt2[0] && seg2_pt1[1] == seg2_pt2[1]) {
    throw std::invalid_argument("Second segment has no length");
  }
  using scalar_t = std::decay_t<decltype(seg1_pt1[0])>;

  const scalar_t seg1_x = seg1_pt2[0] - seg1_pt1[0];
  const scalar_t seg1_y = seg1_pt2[1] - seg1_pt1[1];

  const scalar_t seg2_x = seg2_pt2[0] - seg2_pt1[0];
  const scalar_t seg2_y = seg2_pt2[1] - seg2_pt1[1];

  const scalar_t det = seg1_y * seg2_x - seg1_x * seg2_y;
  if (det == scalar_t(0.)) {
    return false;
  }

  const scalar_t pt1_dx = seg2_pt1[0] - seg1_pt1[0];
  const scalar_t pt1_dy = seg2_pt1[1] - seg1_pt1[1];

  const scalar_t alpha = (pt1_dy * seg2_x - pt1_dx * seg2_y) / det;
  const scalar_t beta = (pt1_dy * seg1_x - pt1_dx * seg1_y) / det;

  if (
    alpha < scalar_t(0.) ||
    alpha > scalar_t(1.) ||
    beta < scalar_t(0.) ||
    beta > scalar_t(1.))
  {
    return false;
  }

  return true;
}

template<typename PointPair>
bool segments_intersect(
  const PointPair & seg1,
  const PointPair & seg2)
{
  return segments_intersect(seg1.first, seg1.second, seg2.first, seg2.second);
}


/***************************************************************************
 * \brief Checks if segment intersects a polygon
 ***************************************************************************/
template<typename ForwardIterator, typename Point>
bool segment_inter_polygon(
  ForwardIterator first,
  ForwardIterator last,
  const Point & point1,
  const Point & point2)
{
  ForwardIterator current = first;
  ForwardIterator next = current; ++next;

  do {
    if (segments_intersect(point1, point2, *current, *next)) {
      return true;
    }

    current = next;
    ++next;
    if (next == last) {next = first;}
  } while (current != first);

  return false;
}

template<typename ForwardIterator, typename PointPair>
bool segment_inter_polygon(
  ForwardIterator first,
  ForwardIterator last,
  const PointPair & seg)
{
  return segment_inter_polygon(first, last, seg.first, seg.second);
}

/***************************************************************************
 * \brief Checks if segment is in the interior of a polygon
 ***************************************************************************/
template<typename ForwardIterator, typename Point>
bool segment_in_polygon(
  ForwardIterator first,
  ForwardIterator last,
  const Point & point1,
  const Point & point2)
{
  if (!point_in_polygon(first, last, point1)) {
    return false;
  }

  return !segment_inter_polygon(first, last, point1, point2);
}

template<typename ForwardIterator, typename PointPair>
bool segment_in_polygon(
  ForwardIterator first,
  ForwardIterator last,
  const PointPair & seg)
{
  return segment_in_polygon(first, last, seg.first, seg.second);
}

/***************************************************************************
 * \brief Array of N evenly spaced numbers
 ***************************************************************************/
template<std::size_t N, typename T>
constexpr std::array<T, N> linspace(T x0, T xT)
{
  static_assert(std::is_floating_point_v<T>, "T must be floating-point number");
  static_assert(N >= 2, "N must be greater or equal to 2");

  T dx = (xT - x0) / static_cast<T>(N - 1);
  std::array<T, N> ret{x0};
  for (std::size_t i = 1; i != N; ++i) {
    ret[i] = x0 += dx;
  }
  return ret;
}

/***************************************************************************
 * \brief Sigmoid with center and steepness
 ***************************************************************************/
template<typename T>
T sigmoid(T x, T center, T k)
{
  static_assert(std::is_floating_point_v<T>, "T must be an arithmetic type");
  using std::exp;

  return T(1) / (T(1) + exp(-k * (x - center)));
}


namespace detail
{
constexpr Eigen::StorageOptions layout(size_t N)
{
  if (N > 1) {
    return Eigen::RowMajor;
  }
  return Eigen::ColMajor;
}
}   // namespace detail

/**
 * Calculate sample error covariance matrix \Sigma for a dataset and an error functions
 * @param data container with samples of type X, e.g. std::vector<X>
 * @param fcn mapping X -> ErrT where ErrT is an Eigen column vector/array of size D
 * @return covariance matrix of size D x D
 *
 *  \Sigma = (1/N-1) * \sum_i (fcn(x_i) - \bar x) * (fcn(x_i) - \bar x).transpose()
 *
 *   where  \bar x = (1/N) * \sum_i fcn(x_i)
 *
 */
template<typename DataContainerT, typename ErrorFcnT>
auto sample_covariance(
  const DataContainerT & data,
  ErrorFcnT && fcn = [](const auto & x) {return x;})
{
  using ResT = typename std::result_of_t<
    ErrorFcnT(typename DataContainerT::value_type)
    >::PlainMatrix;
  static_assert(ResT::ColsAtCompileTime == 1, "fcn must map to column vector");
  static_assert(std::is_base_of_v<Eigen::MatrixBase<ResT>, ResT>, "fcn must map to eigen type");

  static constexpr int N = ResT::RowsAtCompileTime;
  using InfT = Eigen::Matrix<typename ResT::Scalar, N, N>;

  InfT cov = InfT::Zero();

  if (data.size() > 1) {
    using CompT = Eigen::Matrix<
      typename ResT::Scalar, Eigen::Dynamic, N, detail::layout(N)
    >;
    CompT errs(data.size(), N);
    for (size_t i = 0; i != data.size(); ++i) {
      errs.row(i) = fcn(data[i]);
    }

    CompT centered = errs.rowwise() - errs.colwise().mean();
    cov = (centered.adjoint() * centered) / (errs.rows() - 1);
  }

  return cov;
}

/**
 * Calculate sample square root information matrix I = \Sigma^{-1/2}
 * @param cov covariance
 * @param min_eig lower bound for eigenvalues of \Sigma (smaller eigenvalues are set to this value)
 * @return square root information matrix of size D x D
 */
template<typename Derived>
auto sqrt_information(const Eigen::MatrixBase<Derived> & cov, typename Derived::Scalar min_eig)
{
  Eigen::SelfAdjointEigenSolver<typename Derived::PlainMatrix> es(cov);
  return (es.eigenvectors() *
         es.eigenvalues().cwiseMax(min_eig).cwiseSqrt().cwiseInverse().asDiagonal() *
         es.eigenvectors().transpose()).eval();
}

}  // namespace cbr

#endif  // CBR_MATH__MATH_HPP_
