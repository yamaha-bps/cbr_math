/**
  * Original file: geodesic.cpp
  *
  * Copyright (c) Charles Karney (2009-2019) <charles@karney.com> and licensed
  * under the MIT/X11 License.  For more information, see
  * https://geographiclib.sourceforge.io/
  **********************************************************************/

#ifndef CBR_MATH__GEODESIC_HPP_
#define CBR_MATH__GEODESIC_HPP_

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <exception>
#include <utility>
#include <limits>

#include "math.hpp"

#include "geodetic_data.hpp"

#define GEOGRAPHICLIB_GEODESIC_ORDER 6

namespace cbr::geo
{

constexpr static int nA1_ = GEOGRAPHICLIB_GEODESIC_ORDER;
constexpr static int nC1_ = GEOGRAPHICLIB_GEODESIC_ORDER;
constexpr static int nC1p_ = GEOGRAPHICLIB_GEODESIC_ORDER;
constexpr static int nA2_ = GEOGRAPHICLIB_GEODESIC_ORDER;
constexpr static int nC2_ = GEOGRAPHICLIB_GEODESIC_ORDER;
constexpr static int nA3_ = GEOGRAPHICLIB_GEODESIC_ORDER;
constexpr static int nA3x_ = nA3_;
constexpr static int nC3_ = GEOGRAPHICLIB_GEODESIC_ORDER;
constexpr static int nC3x_ = (nC3_ * (nC3_ - 1)) / 2;
constexpr static int nC4_ = GEOGRAPHICLIB_GEODESIC_ORDER;
constexpr static int nC4x_ = (nC4_ * (nC4_ + 1)) / 2;
constexpr static int nC_ = GEOGRAPHICLIB_GEODESIC_ORDER + 1;
static double _A3x[nA3x_], _C3x[nC3x_], _C4x[nC4x_];

enum captype
{
  CAP_NONE = 0U,
  CAP_C1   = 1U << 0,
  CAP_C1p  = 1U << 1,
  CAP_C2   = 1U << 2,
  CAP_C3   = 1U << 3,
  CAP_C4   = 1U << 4,
  CAP_ALL  = 0x1FU,
  CAP_MASK = CAP_ALL,
  OUT_ALL  = 0x7F80U,
  OUT_MASK = 0xFF80U,            // Includes LONG_UNROLL
};

enum mask
{
  /**
   * No capabilities, no output.
   * @hideinitializer
   **********************************************************************/
  NONE          = 0U,
  /**
   * Calculate latitude \e lat2.  (It's not necessary to include this as a
   * capability to GeodesicLine because this is included by default.)
   * @hideinitializer
   **********************************************************************/
  LATITUDE      = 1U << 7 | CAP_NONE,
  /**
   * Calculate longitude \e lon2.
   * @hideinitializer
   **********************************************************************/
  LONGITUDE     = 1U << 8 | CAP_C3,
  /**
   * Calculate azimuths \e azi1 and \e azi2.  (It's not necessary to
   * include this as a capability to GeodesicLine because this is included
   * by default.)
   * @hideinitializer
   **********************************************************************/
  AZIMUTH       = 1U << 9 | CAP_NONE,
  /**
   * Calculate distance \e s12.
   * @hideinitializer
   **********************************************************************/
  DISTANCE      = 1U << 10 | CAP_C1,
  /**
   * Allow distance \e s12 to be used as input in the direct geodesic
   * problem.
   * @hideinitializer
   **********************************************************************/
  DISTANCE_IN   = 1U << 11 | CAP_C1 | CAP_C1p,
  /**
   * Calculate reduced length \e m12.
   * @hideinitializer
   **********************************************************************/
  REDUCEDLENGTH = 1U << 12 | CAP_C1 | CAP_C2,
  /**
   * Calculate geodesic scales \e M12 and \e M21.
   * @hideinitializer
   **********************************************************************/
  GEODESICSCALE = 1U << 13 | CAP_C1 | CAP_C2,
  /**
   * Calculate area \e S12.
   * @hideinitializer
   **********************************************************************/
  AREA          = 1U << 14 | CAP_C4,
  /**
   * Unroll \e lon2 in the direct calculation.
   * @hideinitializer
   **********************************************************************/
  LONG_UNROLL   = 1U << 15,
  /**
   * All capabilities, calculate everything.  (LONG_UNROLL is not
   * included in this mask.)
   * @hideinitializer
   **********************************************************************/
  ALL           = OUT_ALL | CAP_ALL,
};


template<typename T>
T eatanhe(T x, T es)
{
  return es > T(0) ? es * atanh(es * x) : -es * atan(es * x);
}

template<typename T>
constexpr T sq(T x)
{return x * x;}


constexpr inline int digits()
{
  return std::numeric_limits<double>::digits;
}

template<typename T>
T sum(T u, T v, T & t)
{
  volatile T s = u + v;
  volatile T up = s - v;
  volatile T vpp = s - up;
  up -= u;
  vpp -= v;
  t = -(up + vpp);
  // u + v =       s      + t
  //       = round(u + v) + t
  return s;
}

template<typename T>
T polyval(int N, const T p[], T x)
{
  T y = N < 0 ? 0 : *p++; while (--N >= 0) {y = y * x + *p++;} return y;
}

template<typename T>
T AngNormalize(T x)
{
  x = std::remainder(x, T(360));
  return x != -180 ? x : 180;
}

template<typename T>
T AngDiff(T x, T y, T & e)
{
  T t, d = AngNormalize(
    sum(
      std::remainder(-x, T(360)),
      std::remainder(y, T(360)), t));
  return sum(d == 180 && t > 0 ? -180 : d, t, e);
}

template<typename T>
T AngRound(T x)
{
  constexpr T z = T(1) / T(16);
  if (x == 0) {return 0;}
  volatile T y = std::abs(x);
  // The compiler mustn't "simplify" z - (z - y) to y
  y = y < z ? z - (z - y) : y;
  return x < 0 ? -y : y;
}

template<typename T>
void sincosd(T x, T & sinx, T & cosx)
{
  // In order to minimize round-off errors, this function exactly reduces
  // the argument to the range [-45, 45] before converting it to radians.
  T r; int q;
  // N.B. the implementation of std::remquo in glibc pre 2.22 were buggy.  See
  // https://sourceware.org/bugzilla/show_bug.cgi?id=17569
  // This was fixed in version 2.22 on 2015-08-05
  r = std::remquo(x, T(90), &q);      // now abs(r) <= 45
  r *= M_PI / 180.;
  // g++ -O turns these two function calls into a call to sincos
  T s = sin(r), c = cos(r);
  switch (unsigned(q) & 3U) {
    case 0U: sinx = s; cosx = c; break;
    case 1U: sinx = c; cosx = -s; break;
    case 2U: sinx = -s; cosx = -c; break;
    default: sinx = -c; cosx = s; break;   // case 3U
  }
  // Set sign of 0 results.  -0 only produced for sin(-0)
  if (x != 0) {sinx += T(0); cosx += T(0);}
}

template<typename T>
T sind(T x)
{
  // See sincosd
  T r; int q;
  r = std::remquo(x, T(90), &q);    // now abs(r) <= 45
  r *= M_PI / 180.;
  unsigned p = unsigned(q);
  r = p & 1U ? cos(r) : sin(r);
  if (p & 2U) {r = -r;}
  if (x != 0) {r += T(0);}
  return r;
}

template<typename T>
T cosd(T x)
{
  // See sincosd
  T r; int q;
  r = std::remquo(x, T(90), &q);    // now abs(r) <= 45
  r *= M_PI / 180.;
  unsigned p = unsigned(q + 1);
  r = p & 1U ? cos(r) : sin(r);
  if (p & 2U) {r = -r;}
  return T(0) + r;
}

template<typename T>
T tand(T x)
{
  static const T overflow = 1 / sq(std::numeric_limits<T>::epsilon());
  T s, c;
  sincosd(x, s, c);
  return c != 0 ? s / c : (s < 0 ? -overflow : overflow);
}

template<typename T>
void norm(T & x, T & y)
{T h = std::hypot(x, y); x /= h; y /= h;}

template<typename T>
T NaN()
{
  return std::numeric_limits<T>::has_quiet_NaN ?
         std::numeric_limits<T>::quiet_NaN() :
         std::numeric_limits<T>::max();
}

template<typename T>
T LatFix(T x)
{
  return std::abs(x) > 90 ? NaN<T>() : x;
}

template<typename T = double>
T pi()
{
  using std::atan2;
  static const T pi = atan2(T(0), T(-1));
  return pi;
}

inline double SinCosSeries(
  bool sinp,
  double sinx, double cosx,
  const double c[], int n)
{
  // Evaluate
  // y = sinp ? sum(c[i] * sin( 2*i    * x), i, 1, n) :
  //            sum(c[i] * cos((2*i+1) * x), i, 0, n-1)
  // using Clenshaw summation.  N.B. c[0] is unused for sin series
  // Approx operation count = (n + 5) mult and (2 * n + 2) add
  c += (n + sinp);               // Point to one beyond last element
  double ar = 2. * (cosx - sinx) * (cosx + sinx);    // 2 * cos(2 * x)
  double y0 = n & 1 ? *--c : 0.;
  double y1 = 0.;             // accumulators for sum
  // Now n is even
  n /= 2;
  while (n--) {
    // Unroll loop x 2, so accumulators return to their original role
    y1 = ar * y0 - y1 + *--c;
    y0 = ar * y1 - y0 + *--c;
  }
  return sinp ?
         2. * sinx * cosx * y0 :  // sin(2 * x) * y0
         cosx * (y0 - y1);       // cos(x) * (y0 - y1)
}

constexpr static unsigned maxit1_(20);
constexpr static unsigned maxit2_(maxit1_ + digits() + 10);
static const double tiny_(sqrt(std::numeric_limits<double>::min()));
constexpr static double tol0_(std::numeric_limits<double>::epsilon());

constexpr static double tol1_(200 * tol0_);
static const double tol2_(sqrt(tol0_));
static const double tolb_(tol0_ * tol2_);
static const double xthresh_(1000. * tol2_);

inline double Astroid(double x, double y)
{
  // Solve k^4+2*k^3-(x^2+y^2-1)*k^2-2*y^2*k-y^2 = 0 for positive root k.
  // This solution is adapted from Geocentric::Reverse.
  double k;
  double
    p = sq(x),
    q = sq(y),
    r = (p + q - 1) / 6;
  if (!(q == 0 && r <= 0) ) {
    double
    // Avoid possible division by zero when r = 0 by multiplying equations
    // for s and t by r^3 and r, resp.
      S = p * q / 4,               // S = r^3 * s
      r2 = sq(r),
      r3 = r * r2,
    // The discriminant of the quadratic equation for T3.  This is zero on
    // the evolute curve p^(1/3)+q^(1/3) = 1
      disc = S * (S + 2 * r3);
    double u = r;
    if (disc >= 0) {
      double T3 = S + r3;
      // Pick the sign on the sqrt to maximize abs(T3).  This minimizes loss
      // of precision due to cancellation.  The result is unchanged because
      // of the way the T is used in definition of u.
      T3 += T3 < 0 ? -sqrt(disc) : sqrt(disc);    // T3 = (r * t)^3
      // N.B. cbrt always returns the double root.  cbrt(-8) = -2.
      double T = std::cbrt(T3);    // T = r * t
      // T can be zero; but then r2 / T -> 0.
      u += T + (T != 0 ? r2 / T : 0);
    } else {
      // T is complex, but the way u is defined the result is double.
      double ang = atan2(sqrt(-disc), -(S + r3));
      // There are three possible cube roots.  We choose the root which
      // avoids cancellation.  Note that disc < 0 implies that r < 0.
      u += 2 * r * cos(ang / 3);
    }
    double
      v = sqrt(sq(u) + q),       // guaranteed positive
    // Avoid loss of accuracy when u < 0.
      uv = u < 0 ? q / (v - u) : u + v,    // u+v, guaranteed positive
      w = (uv - q) / (2 * v);              // positive?
    // Rearrange expression for k to avoid loss of accuracy due to
    // subtraction.  Division by 0 not possible because uv > 0, w >= 0.
    k = uv / (sqrt(uv + sq(w)) + w);      // guaranteed positive
  } else {                  // q == 0 && r <= 0
    // y = 0 with |x| <= 1.  Handle this case directly.
    // for y small, positive root is k = abs(y)/sqrt(1-x^2)
    k = 0;
  }
  return k;
}

inline double A3f(double eps)
{
  // Evaluate A3
  return polyval(nA3_ - 1, _A3x, eps);
}

inline void C3f(double eps, double c[])
{
  // Evaluate C3 coeffs
  // Elements c[1] thru c[nC3_ - 1] are set
  double mult = 1;
  int o = 0;
  for (int l = 1; l < nC3_; ++l) {    // l is index of C3[l]
    int m = nC3_ - l - 1;             // order of polynomial in eps
    mult *= eps;
    c[l] = mult * polyval(m, _C3x + o, eps);
    o += m + 1;
  }
  // Post condition: o == nC3x_
}

inline void C4f(double eps, double c[])
{
  // Evaluate C4 coeffs
  // Elements c[0] thru c[nC4_ - 1] are set
  double mult = 1;
  int o = 0;
  for (int l = 0; l < nC4_; ++l) {    // l is index of C4[l]
    int m = nC4_ - l - 1;             // order of polynomial in eps
    c[l] = mult * polyval(m, _C4x + o, eps);
    o += m + 1;
    mult *= eps;
  }
  // Post condition: o == nC4x_
}

// The static const coefficient arrays in the following functions are
// generated by Maxima and give the coefficients of the Taylor expansions for
// the geodesics.  The convention on the order of these coefficients is as
// follows:
//
//   ascending order in the trigonometric expansion,
//   then powers of eps in descending order,
//   finally powers of n in descending order.
//
// (For some expansions, only a subset of levels occur.)  For each polynomial
// of order n at the lowest level, the (n+1) coefficients of the polynomial
// are followed by a divisor which is applied to the whole polynomial.  In
// this way, the coefficients are expressible with no round off error.  The
// sizes of the coefficient arrays are:
//
//   A1m1f, A2m1f            = floor(N/2) + 2
//   C1f, C1pf, C2f, A3coeff = (N^2 + 7*N - 2*floor(N/2)) / 4
//   C3coeff       = (N - 1) * (N^2 + 7*N - 2*floor(N/2)) / 8
//   C4coeff       = N * (N + 1) * (N + 5) / 6
//
// where N = GEOGRAPHICLIB_GEODESIC_ORDER
//         = nA1 = nA2 = nC1 = nC1p = nA3 = nC4

// The scale factor A1-1 = mean value of (d/dsigma)I1 - 1
inline double A1m1f(double eps)
{
  // Generated by Maxima on 2015-05-05 18:08:12-04:00
 #if GEOGRAPHICLIB_GEODESIC_ORDER / 2 == 1
  static const double coeff[] = {
    // (1-eps)*A1-1, polynomial in eps2 of order 1
    1, 0, 4,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER / 2 == 2
  static const double coeff[] = {
    // (1-eps)*A1-1, polynomial in eps2 of order 2
    1, 16, 0, 64,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER / 2 == 3
  static const double coeff[] = {
    // (1-eps)*A1-1, polynomial in eps2 of order 3
    1, 4, 64, 0, 256,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER / 2 == 4
  static const double coeff[] = {
    // (1-eps)*A1-1, polynomial in eps2 of order 4
    25, 64, 256, 4096, 0, 16384,
  };
 #else
 #error "Bad value for GEOGRAPHICLIB_GEODESIC_ORDER"
 #endif

  int m = nA1_ / 2;
  double t = polyval(m, coeff, sq(eps)) / coeff[m + 1];
  return (t + eps) / (1 - eps);
}

// The coefficients C1[l] in the Fourier expansion of B1
inline void C1f(double eps, double c[])
{
  // Generated by Maxima on 2015-05-05 18:08:12-04:00
 #if GEOGRAPHICLIB_GEODESIC_ORDER == 3
  static const double coeff[] = {
    // C1[1]/eps^1, polynomial in eps2 of order 1
    3, -8, 16,
    // C1[2]/eps^2, polynomial in eps2 of order 0
    -1, 16,
    // C1[3]/eps^3, polynomial in eps2 of order 0
    -1, 48,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 4
  static const double coeff[] = {
    // C1[1]/eps^1, polynomial in eps2 of order 1
    3, -8, 16,
    // C1[2]/eps^2, polynomial in eps2 of order 1
    1, -2, 32,
    // C1[3]/eps^3, polynomial in eps2 of order 0
    -1, 48,
    // C1[4]/eps^4, polynomial in eps2 of order 0
    -5, 512,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 5
  static const double coeff[] = {
    // C1[1]/eps^1, polynomial in eps2 of order 2
    -1, 6, -16, 32,
    // C1[2]/eps^2, polynomial in eps2 of order 1
    1, -2, 32,
    // C1[3]/eps^3, polynomial in eps2 of order 1
    9, -16, 768,
    // C1[4]/eps^4, polynomial in eps2 of order 0
    -5, 512,
    // C1[5]/eps^5, polynomial in eps2 of order 0
    -7, 1280,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 6
  static const double coeff[] = {
    // C1[1]/eps^1, polynomial in eps2 of order 2
    -1, 6, -16, 32,
    // C1[2]/eps^2, polynomial in eps2 of order 2
    -9, 64, -128, 2048,
    // C1[3]/eps^3, polynomial in eps2 of order 1
    9, -16, 768,
    // C1[4]/eps^4, polynomial in eps2 of order 1
    3, -5, 512,
    // C1[5]/eps^5, polynomial in eps2 of order 0
    -7, 1280,
    // C1[6]/eps^6, polynomial in eps2 of order 0
    -7, 2048,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 7
  static const double coeff[] = {
    // C1[1]/eps^1, polynomial in eps2 of order 3
    19, -64, 384, -1024, 2048,
    // C1[2]/eps^2, polynomial in eps2 of order 2
    -9, 64, -128, 2048,
    // C1[3]/eps^3, polynomial in eps2 of order 2
    -9, 72, -128, 6144,
    // C1[4]/eps^4, polynomial in eps2 of order 1
    3, -5, 512,
    // C1[5]/eps^5, polynomial in eps2 of order 1
    35, -56, 10240,
    // C1[6]/eps^6, polynomial in eps2 of order 0
    -7, 2048,
    // C1[7]/eps^7, polynomial in eps2 of order 0
    -33, 14336,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 8
  static const double coeff[] = {
    // C1[1]/eps^1, polynomial in eps2 of order 3
    19, -64, 384, -1024, 2048,
    // C1[2]/eps^2, polynomial in eps2 of order 3
    7, -18, 128, -256, 4096,
    // C1[3]/eps^3, polynomial in eps2 of order 2
    -9, 72, -128, 6144,
    // C1[4]/eps^4, polynomial in eps2 of order 2
    -11, 96, -160, 16384,
    // C1[5]/eps^5, polynomial in eps2 of order 1
    35, -56, 10240,
    // C1[6]/eps^6, polynomial in eps2 of order 1
    9, -14, 4096,
    // C1[7]/eps^7, polynomial in eps2 of order 0
    -33, 14336,
    // C1[8]/eps^8, polynomial in eps2 of order 0
    -429, 262144,
  };
 #else
 #error "Bad value for GEOGRAPHICLIB_GEODESIC_ORDER"
 #endif
  double
    eps2 = sq(eps),
    d = eps;
  int o = 0;
  for (int l = 1; l <= nC1_; ++l) {    // l is index of C1p[l]
    int m = (nC1_ - l) / 2;            // order of polynomial in eps^2
    c[l] = d * polyval(m, coeff + o, eps2) / coeff[o + m + 1];
    o += m + 2;
    d *= eps;
  }
  // Post condition: o == sizeof(coeff) / sizeof(double)
}

// The coefficients C1p[l] in the Fourier expansion of B1p
inline void C1pf(double eps, double c[])
{
  // Generated by Maxima on 2015-05-05 18:08:12-04:00
 #if GEOGRAPHICLIB_GEODESIC_ORDER == 3
  static const double coeff[] = {
    // C1p[1]/eps^1, polynomial in eps2 of order 1
    -9, 16, 32,
    // C1p[2]/eps^2, polynomial in eps2 of order 0
    5, 16,
    // C1p[3]/eps^3, polynomial in eps2 of order 0
    29, 96,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 4
  static const double coeff[] = {
    // C1p[1]/eps^1, polynomial in eps2 of order 1
    -9, 16, 32,
    // C1p[2]/eps^2, polynomial in eps2 of order 1
    -37, 30, 96,
    // C1p[3]/eps^3, polynomial in eps2 of order 0
    29, 96,
    // C1p[4]/eps^4, polynomial in eps2 of order 0
    539, 1536,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 5
  static const double coeff[] = {
    // C1p[1]/eps^1, polynomial in eps2 of order 2
    205, -432, 768, 1536,
    // C1p[2]/eps^2, polynomial in eps2 of order 1
    -37, 30, 96,
    // C1p[3]/eps^3, polynomial in eps2 of order 1
    -225, 116, 384,
    // C1p[4]/eps^4, polynomial in eps2 of order 0
    539, 1536,
    // C1p[5]/eps^5, polynomial in eps2 of order 0
    3467, 7680,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 6
  static const double coeff[] = {
    // C1p[1]/eps^1, polynomial in eps2 of order 2
    205, -432, 768, 1536,
    // C1p[2]/eps^2, polynomial in eps2 of order 2
    4005, -4736, 3840, 12288,
    // C1p[3]/eps^3, polynomial in eps2 of order 1
    -225, 116, 384,
    // C1p[4]/eps^4, polynomial in eps2 of order 1
    -7173, 2695, 7680,
    // C1p[5]/eps^5, polynomial in eps2 of order 0
    3467, 7680,
    // C1p[6]/eps^6, polynomial in eps2 of order 0
    38081, 61440,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 7
  static const double coeff[] = {
    // C1p[1]/eps^1, polynomial in eps2 of order 3
    -4879, 9840, -20736, 36864, 73728,
    // C1p[2]/eps^2, polynomial in eps2 of order 2
    4005, -4736, 3840, 12288,
    // C1p[3]/eps^3, polynomial in eps2 of order 2
    8703, -7200, 3712, 12288,
    // C1p[4]/eps^4, polynomial in eps2 of order 1
    -7173, 2695, 7680,
    // C1p[5]/eps^5, polynomial in eps2 of order 1
    -141115, 41604, 92160,
    // C1p[6]/eps^6, polynomial in eps2 of order 0
    38081, 61440,
    // C1p[7]/eps^7, polynomial in eps2 of order 0
    459485, 516096,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 8
  static const double coeff[] = {
    // C1p[1]/eps^1, polynomial in eps2 of order 3
    -4879, 9840, -20736, 36864, 73728,
    // C1p[2]/eps^2, polynomial in eps2 of order 3
    -86171, 120150, -142080, 115200, 368640,
    // C1p[3]/eps^3, polynomial in eps2 of order 2
    8703, -7200, 3712, 12288,
    // C1p[4]/eps^4, polynomial in eps2 of order 2
    1082857, -688608, 258720, 737280,
    // C1p[5]/eps^5, polynomial in eps2 of order 1
    -141115, 41604, 92160,
    // C1p[6]/eps^6, polynomial in eps2 of order 1
    -2200311, 533134, 860160,
    // C1p[7]/eps^7, polynomial in eps2 of order 0
    459485, 516096,
    // C1p[8]/eps^8, polynomial in eps2 of order 0
    109167851, 82575360,
  };
 #else
 #error "Bad value for GEOGRAPHICLIB_GEODESIC_ORDER"
 #endif
  double
    eps2 = sq(eps),
    d = eps;
  int o = 0;
  for (int l = 1; l <= nC1p_; ++l) {    // l is index of C1p[l]
    int m = (nC1p_ - l) / 2;            // order of polynomial in eps^2
    c[l] = d * polyval(m, coeff + o, eps2) / coeff[o + m + 1];
    o += m + 2;
    d *= eps;
  }
  // Post condition: o == sizeof(coeff) / sizeof(double)
}

// The scale factor A2-1 = mean value of (d/dsigma)I2 - 1
inline double A2m1f(double eps)
{
  // Generated by Maxima on 2015-05-29 08:09:47-04:00
 #if GEOGRAPHICLIB_GEODESIC_ORDER / 2 == 1
  static const double coeff[] = {
    // (eps+1)*A2-1, polynomial in eps2 of order 1
    -3, 0, 4,
  };     // count = 3
 #elif GEOGRAPHICLIB_GEODESIC_ORDER / 2 == 2
  static const double coeff[] = {
    // (eps+1)*A2-1, polynomial in eps2 of order 2
    -7, -48, 0, 64,
  };     // count = 4
 #elif GEOGRAPHICLIB_GEODESIC_ORDER / 2 == 3
  static const double coeff[] = {
    // (eps+1)*A2-1, polynomial in eps2 of order 3
    -11, -28, -192, 0, 256,
  };     // count = 5
 #elif GEOGRAPHICLIB_GEODESIC_ORDER / 2 == 4
  static const double coeff[] = {
    // (eps+1)*A2-1, polynomial in eps2 of order 4
    -375, -704, -1792, -12288, 0, 16384,
  };     // count = 6
 #else
 #error "Bad value for GEOGRAPHICLIB_GEODESIC_ORDER"
 #endif
  int m = nA2_ / 2;
  double t = polyval(m, coeff, sq(eps)) / coeff[m + 1];
  return (t - eps) / (1 + eps);
}

// The coefficients C2[l] in the Fourier expansion of B2
inline void C2f(double eps, double c[])
{
  // Generated by Maxima on 2015-05-05 18:08:12-04:00
 #if GEOGRAPHICLIB_GEODESIC_ORDER == 3
  static const double coeff[] = {
    // C2[1]/eps^1, polynomial in eps2 of order 1
    1, 8, 16,
    // C2[2]/eps^2, polynomial in eps2 of order 0
    3, 16,
    // C2[3]/eps^3, polynomial in eps2 of order 0
    5, 48,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 4
  static const double coeff[] = {
    // C2[1]/eps^1, polynomial in eps2 of order 1
    1, 8, 16,
    // C2[2]/eps^2, polynomial in eps2 of order 1
    1, 6, 32,
    // C2[3]/eps^3, polynomial in eps2 of order 0
    5, 48,
    // C2[4]/eps^4, polynomial in eps2 of order 0
    35, 512,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 5
  static const double coeff[] = {
    // C2[1]/eps^1, polynomial in eps2 of order 2
    1, 2, 16, 32,
    // C2[2]/eps^2, polynomial in eps2 of order 1
    1, 6, 32,
    // C2[3]/eps^3, polynomial in eps2 of order 1
    15, 80, 768,
    // C2[4]/eps^4, polynomial in eps2 of order 0
    35, 512,
    // C2[5]/eps^5, polynomial in eps2 of order 0
    63, 1280,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 6
  static const double coeff[] = {
    // C2[1]/eps^1, polynomial in eps2 of order 2
    1, 2, 16, 32,
    // C2[2]/eps^2, polynomial in eps2 of order 2
    35, 64, 384, 2048,
    // C2[3]/eps^3, polynomial in eps2 of order 1
    15, 80, 768,
    // C2[4]/eps^4, polynomial in eps2 of order 1
    7, 35, 512,
    // C2[5]/eps^5, polynomial in eps2 of order 0
    63, 1280,
    // C2[6]/eps^6, polynomial in eps2 of order 0
    77, 2048,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 7
  static const double coeff[] = {
    // C2[1]/eps^1, polynomial in eps2 of order 3
    41, 64, 128, 1024, 2048,
    // C2[2]/eps^2, polynomial in eps2 of order 2
    35, 64, 384, 2048,
    // C2[3]/eps^3, polynomial in eps2 of order 2
    69, 120, 640, 6144,
    // C2[4]/eps^4, polynomial in eps2 of order 1
    7, 35, 512,
    // C2[5]/eps^5, polynomial in eps2 of order 1
    105, 504, 10240,
    // C2[6]/eps^6, polynomial in eps2 of order 0
    77, 2048,
    // C2[7]/eps^7, polynomial in eps2 of order 0
    429, 14336,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 8
  static const double coeff[] = {
    // C2[1]/eps^1, polynomial in eps2 of order 3
    41, 64, 128, 1024, 2048,
    // C2[2]/eps^2, polynomial in eps2 of order 3
    47, 70, 128, 768, 4096,
    // C2[3]/eps^3, polynomial in eps2 of order 2
    69, 120, 640, 6144,
    // C2[4]/eps^4, polynomial in eps2 of order 2
    133, 224, 1120, 16384,
    // C2[5]/eps^5, polynomial in eps2 of order 1
    105, 504, 10240,
    // C2[6]/eps^6, polynomial in eps2 of order 1
    33, 154, 4096,
    // C2[7]/eps^7, polynomial in eps2 of order 0
    429, 14336,
    // C2[8]/eps^8, polynomial in eps2 of order 0
    6435, 262144,
  };
 #else
 #error "Bad value for GEOGRAPHICLIB_GEODESIC_ORDER"
 #endif
  double
    eps2 = sq(eps),
    d = eps;
  int o = 0;
  for (int l = 1; l <= nC2_; ++l) {    // l is index of C2[l]
    int m = (nC2_ - l) / 2;            // order of polynomial in eps^2
    c[l] = d * polyval(m, coeff + o, eps2) / coeff[o + m + 1];
    o += m + 2;
    d *= eps;
  }
  // Post condition: o == sizeof(coeff) / sizeof(double)
}

// The scale factor A3 = mean value of (d/dsigma)I3
template<typename _ref = WGS84>
void A3coeff()
{
  constexpr double _n = _ref::f / ( 2. - _ref::f);
  // Generated by Maxima on 2015-05-05 18:08:13-04:00
 #if GEOGRAPHICLIB_GEODESIC_ORDER == 3
  static const double coeff[] = {
    // A3, coeff of eps^2, polynomial in n of order 0
    -1, 4,
    // A3, coeff of eps^1, polynomial in n of order 1
    1, -1, 2,
    // A3, coeff of eps^0, polynomial in n of order 0
    1, 1,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 4
  static const double coeff[] = {
    // A3, coeff of eps^3, polynomial in n of order 0
    -1, 16,
    // A3, coeff of eps^2, polynomial in n of order 1
    -1, -2, 8,
    // A3, coeff of eps^1, polynomial in n of order 1
    1, -1, 2,
    // A3, coeff of eps^0, polynomial in n of order 0
    1, 1,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 5
  static const double coeff[] = {
    // A3, coeff of eps^4, polynomial in n of order 0
    -3, 64,
    // A3, coeff of eps^3, polynomial in n of order 1
    -3, -1, 16,
    // A3, coeff of eps^2, polynomial in n of order 2
    3, -1, -2, 8,
    // A3, coeff of eps^1, polynomial in n of order 1
    1, -1, 2,
    // A3, coeff of eps^0, polynomial in n of order 0
    1, 1,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 6
  static const double coeff[] = {
    // A3, coeff of eps^5, polynomial in n of order 0
    -3, 128,
    // A3, coeff of eps^4, polynomial in n of order 1
    -2, -3, 64,
    // A3, coeff of eps^3, polynomial in n of order 2
    -1, -3, -1, 16,
    // A3, coeff of eps^2, polynomial in n of order 2
    3, -1, -2, 8,
    // A3, coeff of eps^1, polynomial in n of order 1
    1, -1, 2,
    // A3, coeff of eps^0, polynomial in n of order 0
    1, 1,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 7
  static const double coeff[] = {
    // A3, coeff of eps^6, polynomial in n of order 0
    -5, 256,
    // A3, coeff of eps^5, polynomial in n of order 1
    -5, -3, 128,
    // A3, coeff of eps^4, polynomial in n of order 2
    -10, -2, -3, 64,
    // A3, coeff of eps^3, polynomial in n of order 3
    5, -1, -3, -1, 16,
    // A3, coeff of eps^2, polynomial in n of order 2
    3, -1, -2, 8,
    // A3, coeff of eps^1, polynomial in n of order 1
    1, -1, 2,
    // A3, coeff of eps^0, polynomial in n of order 0
    1, 1,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 8
  static const double coeff[] = {
    // A3, coeff of eps^7, polynomial in n of order 0
    -25, 2048,
    // A3, coeff of eps^6, polynomial in n of order 1
    -15, -20, 1024,
    // A3, coeff of eps^5, polynomial in n of order 2
    -5, -10, -6, 256,
    // A3, coeff of eps^4, polynomial in n of order 3
    -5, -20, -4, -6, 128,
    // A3, coeff of eps^3, polynomial in n of order 3
    5, -1, -3, -1, 16,
    // A3, coeff of eps^2, polynomial in n of order 2
    3, -1, -2, 8,
    // A3, coeff of eps^1, polynomial in n of order 1
    1, -1, 2,
    // A3, coeff of eps^0, polynomial in n of order 0
    1, 1,
  };
 #else
 #error "Bad value for GEOGRAPHICLIB_GEODESIC_ORDER"
 #endif
  int o = 0, k = 0;
  for (int j = nA3_ - 1; j >= 0; --j) {    // coeff of eps^j
    int m = std::min(nA3_ - j - 1, j);          // order of polynomial in n
    _A3x[k++] = polyval(m, coeff + o, _n) / coeff[o + m + 1];
    o += m + 2;
  }
  // Post condition: o == sizeof(coeff) / sizeof(double) && k == nA3x_
}

// The coefficients C3[l] in the Fourier expansion of B3
template<typename _ref = WGS84>
void C3coeff()
{
  constexpr double _n = _ref::f / ( 2. - _ref::f);
  // Generated by Maxima on 2015-05-05 18:08:13-04:00
 #if GEOGRAPHICLIB_GEODESIC_ORDER == 3
  static const double coeff[] = {
    // C3[1], coeff of eps^2, polynomial in n of order 0
    1, 8,
    // C3[1], coeff of eps^1, polynomial in n of order 1
    -1, 1, 4,
    // C3[2], coeff of eps^2, polynomial in n of order 0
    1, 16,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 4
  static const double coeff[] = {
    // C3[1], coeff of eps^3, polynomial in n of order 0
    3, 64,
    // C3[1], coeff of eps^2, polynomial in n of order 1
    // This is a case where a leading 0 term has been inserted to maintain the
    // pattern in the orders of the polynomials.
    0, 1, 8,
    // C3[1], coeff of eps^1, polynomial in n of order 1
    -1, 1, 4,
    // C3[2], coeff of eps^3, polynomial in n of order 0
    3, 64,
    // C3[2], coeff of eps^2, polynomial in n of order 1
    -3, 2, 32,
    // C3[3], coeff of eps^3, polynomial in n of order 0
    5, 192,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 5
  static const double coeff[] = {
    // C3[1], coeff of eps^4, polynomial in n of order 0
    5, 128,
    // C3[1], coeff of eps^3, polynomial in n of order 1
    3, 3, 64,
    // C3[1], coeff of eps^2, polynomial in n of order 2
    -1, 0, 1, 8,
    // C3[1], coeff of eps^1, polynomial in n of order 1
    -1, 1, 4,
    // C3[2], coeff of eps^4, polynomial in n of order 0
    3, 128,
    // C3[2], coeff of eps^3, polynomial in n of order 1
    -2, 3, 64,
    // C3[2], coeff of eps^2, polynomial in n of order 2
    1, -3, 2, 32,
    // C3[3], coeff of eps^4, polynomial in n of order 0
    3, 128,
    // C3[3], coeff of eps^3, polynomial in n of order 1
    -9, 5, 192,
    // C3[4], coeff of eps^4, polynomial in n of order 0
    7, 512,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 6
  static const double coeff[] = {
    // C3[1], coeff of eps^5, polynomial in n of order 0
    3, 128,
    // C3[1], coeff of eps^4, polynomial in n of order 1
    2, 5, 128,
    // C3[1], coeff of eps^3, polynomial in n of order 2
    -1, 3, 3, 64,
    // C3[1], coeff of eps^2, polynomial in n of order 2
    -1, 0, 1, 8,
    // C3[1], coeff of eps^1, polynomial in n of order 1
    -1, 1, 4,
    // C3[2], coeff of eps^5, polynomial in n of order 0
    5, 256,
    // C3[2], coeff of eps^4, polynomial in n of order 1
    1, 3, 128,
    // C3[2], coeff of eps^3, polynomial in n of order 2
    -3, -2, 3, 64,
    // C3[2], coeff of eps^2, polynomial in n of order 2
    1, -3, 2, 32,
    // C3[3], coeff of eps^5, polynomial in n of order 0
    7, 512,
    // C3[3], coeff of eps^4, polynomial in n of order 1
    -10, 9, 384,
    // C3[3], coeff of eps^3, polynomial in n of order 2
    5, -9, 5, 192,
    // C3[4], coeff of eps^5, polynomial in n of order 0
    7, 512,
    // C3[4], coeff of eps^4, polynomial in n of order 1
    -14, 7, 512,
    // C3[5], coeff of eps^5, polynomial in n of order 0
    21, 2560,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 7
  static const double coeff[] = {
    // C3[1], coeff of eps^6, polynomial in n of order 0
    21, 1024,
    // C3[1], coeff of eps^5, polynomial in n of order 1
    11, 12, 512,
    // C3[1], coeff of eps^4, polynomial in n of order 2
    2, 2, 5, 128,
    // C3[1], coeff of eps^3, polynomial in n of order 3
    -5, -1, 3, 3, 64,
    // C3[1], coeff of eps^2, polynomial in n of order 2
    -1, 0, 1, 8,
    // C3[1], coeff of eps^1, polynomial in n of order 1
    -1, 1, 4,
    // C3[2], coeff of eps^6, polynomial in n of order 0
    27, 2048,
    // C3[2], coeff of eps^5, polynomial in n of order 1
    1, 5, 256,
    // C3[2], coeff of eps^4, polynomial in n of order 2
    -9, 2, 6, 256,
    // C3[2], coeff of eps^3, polynomial in n of order 3
    2, -3, -2, 3, 64,
    // C3[2], coeff of eps^2, polynomial in n of order 2
    1, -3, 2, 32,
    // C3[3], coeff of eps^6, polynomial in n of order 0
    3, 256,
    // C3[3], coeff of eps^5, polynomial in n of order 1
    -4, 21, 1536,
    // C3[3], coeff of eps^4, polynomial in n of order 2
    -6, -10, 9, 384,
    // C3[3], coeff of eps^3, polynomial in n of order 3
    -1, 5, -9, 5, 192,
    // C3[4], coeff of eps^6, polynomial in n of order 0
    9, 1024,
    // C3[4], coeff of eps^5, polynomial in n of order 1
    -10, 7, 512,
    // C3[4], coeff of eps^4, polynomial in n of order 2
    10, -14, 7, 512,
    // C3[5], coeff of eps^6, polynomial in n of order 0
    9, 1024,
    // C3[5], coeff of eps^5, polynomial in n of order 1
    -45, 21, 2560,
    // C3[6], coeff of eps^6, polynomial in n of order 0
    11, 2048,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 8
  static const double coeff[] = {
    // C3[1], coeff of eps^7, polynomial in n of order 0
    243, 16384,
    // C3[1], coeff of eps^6, polynomial in n of order 1
    10, 21, 1024,
    // C3[1], coeff of eps^5, polynomial in n of order 2
    3, 11, 12, 512,
    // C3[1], coeff of eps^4, polynomial in n of order 3
    -2, 2, 2, 5, 128,
    // C3[1], coeff of eps^3, polynomial in n of order 3
    -5, -1, 3, 3, 64,
    // C3[1], coeff of eps^2, polynomial in n of order 2
    -1, 0, 1, 8,
    // C3[1], coeff of eps^1, polynomial in n of order 1
    -1, 1, 4,
    // C3[2], coeff of eps^7, polynomial in n of order 0
    187, 16384,
    // C3[2], coeff of eps^6, polynomial in n of order 1
    69, 108, 8192,
    // C3[2], coeff of eps^5, polynomial in n of order 2
    -2, 1, 5, 256,
    // C3[2], coeff of eps^4, polynomial in n of order 3
    -6, -9, 2, 6, 256,
    // C3[2], coeff of eps^3, polynomial in n of order 3
    2, -3, -2, 3, 64,
    // C3[2], coeff of eps^2, polynomial in n of order 2
    1, -3, 2, 32,
    // C3[3], coeff of eps^7, polynomial in n of order 0
    139, 16384,
    // C3[3], coeff of eps^6, polynomial in n of order 1
    -1, 12, 1024,
    // C3[3], coeff of eps^5, polynomial in n of order 2
    -77, -8, 42, 3072,
    // C3[3], coeff of eps^4, polynomial in n of order 3
    10, -6, -10, 9, 384,
    // C3[3], coeff of eps^3, polynomial in n of order 3
    -1, 5, -9, 5, 192,
    // C3[4], coeff of eps^7, polynomial in n of order 0
    127, 16384,
    // C3[4], coeff of eps^6, polynomial in n of order 1
    -43, 72, 8192,
    // C3[4], coeff of eps^5, polynomial in n of order 2
    -7, -40, 28, 2048,
    // C3[4], coeff of eps^4, polynomial in n of order 3
    -7, 20, -28, 14, 1024,
    // C3[5], coeff of eps^7, polynomial in n of order 0
    99, 16384,
    // C3[5], coeff of eps^6, polynomial in n of order 1
    -15, 9, 1024,
    // C3[5], coeff of eps^5, polynomial in n of order 2
    75, -90, 42, 5120,
    // C3[6], coeff of eps^7, polynomial in n of order 0
    99, 16384,
    // C3[6], coeff of eps^6, polynomial in n of order 1
    -99, 44, 8192,
    // C3[7], coeff of eps^7, polynomial in n of order 0
    429, 114688,
  };
 #else
 #error "Bad value for GEOGRAPHICLIB_GEODESIC_ORDER"
 #endif
  int o = 0, k = 0;
  for (int l = 1; l < nC3_; ++l) {           // l is index of C3[l]
    for (int j = nC3_ - 1; j >= l; --j) {    // coeff of eps^j
      int m = std::min(nC3_ - j - 1, j);          // order of polynomial in n
      _C3x[k++] = polyval(m, coeff + o, _n) / coeff[o + m + 1];
      o += m + 2;
    }
  }
  // Post condition: o == sizeof(coeff) / sizeof(double) && k == nC3x_
}

template<typename _ref = WGS84>
void C4coeff()
{
  constexpr double _n = _ref::f / ( 2. - _ref::f);
  // Generated by Maxima on 2015-05-05 18:08:13-04:00
 #if GEOGRAPHICLIB_GEODESIC_ORDER == 3
  static const double coeff[] = {
    // C4[0], coeff of eps^2, polynomial in n of order 0
    -2, 105,
    // C4[0], coeff of eps^1, polynomial in n of order 1
    16, -7, 35,
    // C4[0], coeff of eps^0, polynomial in n of order 2
    8, -28, 70, 105,
    // C4[1], coeff of eps^2, polynomial in n of order 0
    -2, 105,
    // C4[1], coeff of eps^1, polynomial in n of order 1
    -16, 7, 315,
    // C4[2], coeff of eps^2, polynomial in n of order 0
    4, 525,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 4
  static const double coeff[] = {
    // C4[0], coeff of eps^3, polynomial in n of order 0
    11, 315,
    // C4[0], coeff of eps^2, polynomial in n of order 1
    -32, -6, 315,
    // C4[0], coeff of eps^1, polynomial in n of order 2
    -32, 48, -21, 105,
    // C4[0], coeff of eps^0, polynomial in n of order 3
    4, 24, -84, 210, 315,
    // C4[1], coeff of eps^3, polynomial in n of order 0
    -1, 105,
    // C4[1], coeff of eps^2, polynomial in n of order 1
    64, -18, 945,
    // C4[1], coeff of eps^1, polynomial in n of order 2
    32, -48, 21, 945,
    // C4[2], coeff of eps^3, polynomial in n of order 0
    -8, 1575,
    // C4[2], coeff of eps^2, polynomial in n of order 1
    -32, 12, 1575,
    // C4[3], coeff of eps^3, polynomial in n of order 0
    8, 2205,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 5
  static const double coeff[] = {
    // C4[0], coeff of eps^4, polynomial in n of order 0
    4, 1155,
    // C4[0], coeff of eps^3, polynomial in n of order 1
    -368, 121, 3465,
    // C4[0], coeff of eps^2, polynomial in n of order 2
    1088, -352, -66, 3465,
    // C4[0], coeff of eps^1, polynomial in n of order 3
    48, -352, 528, -231, 1155,
    // C4[0], coeff of eps^0, polynomial in n of order 4
    16, 44, 264, -924, 2310, 3465,
    // C4[1], coeff of eps^4, polynomial in n of order 0
    4, 1155,
    // C4[1], coeff of eps^3, polynomial in n of order 1
    80, -99, 10395,
    // C4[1], coeff of eps^2, polynomial in n of order 2
    -896, 704, -198, 10395,
    // C4[1], coeff of eps^1, polynomial in n of order 3
    -48, 352, -528, 231, 10395,
    // C4[2], coeff of eps^4, polynomial in n of order 0
    -8, 1925,
    // C4[2], coeff of eps^3, polynomial in n of order 1
    384, -88, 17325,
    // C4[2], coeff of eps^2, polynomial in n of order 2
    320, -352, 132, 17325,
    // C4[3], coeff of eps^4, polynomial in n of order 0
    -16, 8085,
    // C4[3], coeff of eps^3, polynomial in n of order 1
    -256, 88, 24255,
    // C4[4], coeff of eps^4, polynomial in n of order 0
    64, 31185,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 6
  static const double coeff[] = {
    // C4[0], coeff of eps^5, polynomial in n of order 0
    97, 15015,
    // C4[0], coeff of eps^4, polynomial in n of order 1
    1088, 156, 45045,
    // C4[0], coeff of eps^3, polynomial in n of order 2
    -224, -4784, 1573, 45045,
    // C4[0], coeff of eps^2, polynomial in n of order 3
    -10656, 14144, -4576, -858, 45045,
    // C4[0], coeff of eps^1, polynomial in n of order 4
    64, 624, -4576, 6864, -3003, 15015,
    // C4[0], coeff of eps^0, polynomial in n of order 5
    100, 208, 572, 3432, -12012, 30030, 45045,
    // C4[1], coeff of eps^5, polynomial in n of order 0
    1, 9009,
    // C4[1], coeff of eps^4, polynomial in n of order 1
    -2944, 468, 135135,
    // C4[1], coeff of eps^3, polynomial in n of order 2
    5792, 1040, -1287, 135135,
    // C4[1], coeff of eps^2, polynomial in n of order 3
    5952, -11648, 9152, -2574, 135135,
    // C4[1], coeff of eps^1, polynomial in n of order 4
    -64, -624, 4576, -6864, 3003, 135135,
    // C4[2], coeff of eps^5, polynomial in n of order 0
    8, 10725,
    // C4[2], coeff of eps^4, polynomial in n of order 1
    1856, -936, 225225,
    // C4[2], coeff of eps^3, polynomial in n of order 2
    -8448, 4992, -1144, 225225,
    // C4[2], coeff of eps^2, polynomial in n of order 3
    -1440, 4160, -4576, 1716, 225225,
    // C4[3], coeff of eps^5, polynomial in n of order 0
    -136, 63063,
    // C4[3], coeff of eps^4, polynomial in n of order 1
    1024, -208, 105105,
    // C4[3], coeff of eps^3, polynomial in n of order 2
    3584, -3328, 1144, 315315,
    // C4[4], coeff of eps^5, polynomial in n of order 0
    -128, 135135,
    // C4[4], coeff of eps^4, polynomial in n of order 1
    -2560, 832, 405405,
    // C4[5], coeff of eps^5, polynomial in n of order 0
    128, 99099,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 7
  static const double coeff[] = {
    // C4[0], coeff of eps^6, polynomial in n of order 0
    10, 9009,
    // C4[0], coeff of eps^5, polynomial in n of order 1
    -464, 291, 45045,
    // C4[0], coeff of eps^4, polynomial in n of order 2
    -4480, 1088, 156, 45045,
    // C4[0], coeff of eps^3, polynomial in n of order 3
    10736, -224, -4784, 1573, 45045,
    // C4[0], coeff of eps^2, polynomial in n of order 4
    1664, -10656, 14144, -4576, -858, 45045,
    // C4[0], coeff of eps^1, polynomial in n of order 5
    16, 64, 624, -4576, 6864, -3003, 15015,
    // C4[0], coeff of eps^0, polynomial in n of order 6
    56, 100, 208, 572, 3432, -12012, 30030, 45045,
    // C4[1], coeff of eps^6, polynomial in n of order 0
    10, 9009,
    // C4[1], coeff of eps^5, polynomial in n of order 1
    112, 15, 135135,
    // C4[1], coeff of eps^4, polynomial in n of order 2
    3840, -2944, 468, 135135,
    // C4[1], coeff of eps^3, polynomial in n of order 3
    -10704, 5792, 1040, -1287, 135135,
    // C4[1], coeff of eps^2, polynomial in n of order 4
    -768, 5952, -11648, 9152, -2574, 135135,
    // C4[1], coeff of eps^1, polynomial in n of order 5
    -16, -64, -624, 4576, -6864, 3003, 135135,
    // C4[2], coeff of eps^6, polynomial in n of order 0
    -4, 25025,
    // C4[2], coeff of eps^5, polynomial in n of order 1
    -1664, 168, 225225,
    // C4[2], coeff of eps^4, polynomial in n of order 2
    1664, 1856, -936, 225225,
    // C4[2], coeff of eps^3, polynomial in n of order 3
    6784, -8448, 4992, -1144, 225225,
    // C4[2], coeff of eps^2, polynomial in n of order 4
    128, -1440, 4160, -4576, 1716, 225225,
    // C4[3], coeff of eps^6, polynomial in n of order 0
    64, 315315,
    // C4[3], coeff of eps^5, polynomial in n of order 1
    1792, -680, 315315,
    // C4[3], coeff of eps^4, polynomial in n of order 2
    -2048, 1024, -208, 105105,
    // C4[3], coeff of eps^3, polynomial in n of order 3
    -1792, 3584, -3328, 1144, 315315,
    // C4[4], coeff of eps^6, polynomial in n of order 0
    -512, 405405,
    // C4[4], coeff of eps^5, polynomial in n of order 1
    2048, -384, 405405,
    // C4[4], coeff of eps^4, polynomial in n of order 2
    3072, -2560, 832, 405405,
    // C4[5], coeff of eps^6, polynomial in n of order 0
    -256, 495495,
    // C4[5], coeff of eps^5, polynomial in n of order 1
    -2048, 640, 495495,
    // C4[6], coeff of eps^6, polynomial in n of order 0
    512, 585585,
  };
 #elif GEOGRAPHICLIB_GEODESIC_ORDER == 8
  static const double coeff[] = {
    // C4[0], coeff of eps^7, polynomial in n of order 0
    193, 85085,
    // C4[0], coeff of eps^6, polynomial in n of order 1
    4192, 850, 765765,
    // C4[0], coeff of eps^5, polynomial in n of order 2
    20960, -7888, 4947, 765765,
    // C4[0], coeff of eps^4, polynomial in n of order 3
    12480, -76160, 18496, 2652, 765765,
    // C4[0], coeff of eps^3, polynomial in n of order 4
    -154048, 182512, -3808, -81328, 26741, 765765,
    // C4[0], coeff of eps^2, polynomial in n of order 5
    3232, 28288, -181152, 240448, -77792, -14586, 765765,
    // C4[0], coeff of eps^1, polynomial in n of order 6
    96, 272, 1088, 10608, -77792, 116688, -51051, 255255,
    // C4[0], coeff of eps^0, polynomial in n of order 7
    588, 952, 1700, 3536, 9724, 58344, -204204, 510510, 765765,
    // C4[1], coeff of eps^7, polynomial in n of order 0
    349, 2297295,
    // C4[1], coeff of eps^6, polynomial in n of order 1
    -1472, 510, 459459,
    // C4[1], coeff of eps^5, polynomial in n of order 2
    -39840, 1904, 255, 2297295,
    // C4[1], coeff of eps^4, polynomial in n of order 3
    52608, 65280, -50048, 7956, 2297295,
    // C4[1], coeff of eps^3, polynomial in n of order 4
    103744, -181968, 98464, 17680, -21879, 2297295,
    // C4[1], coeff of eps^2, polynomial in n of order 5
    -1344, -13056, 101184, -198016, 155584, -43758, 2297295,
    // C4[1], coeff of eps^1, polynomial in n of order 6
    -96, -272, -1088, -10608, 77792, -116688, 51051, 2297295,
    // C4[2], coeff of eps^7, polynomial in n of order 0
    464, 1276275,
    // C4[2], coeff of eps^6, polynomial in n of order 1
    -928, -612, 3828825,
    // C4[2], coeff of eps^5, polynomial in n of order 2
    64256, -28288, 2856, 3828825,
    // C4[2], coeff of eps^4, polynomial in n of order 3
    -126528, 28288, 31552, -15912, 3828825,
    // C4[2], coeff of eps^3, polynomial in n of order 4
    -41472, 115328, -143616, 84864, -19448, 3828825,
    // C4[2], coeff of eps^2, polynomial in n of order 5
    160, 2176, -24480, 70720, -77792, 29172, 3828825,
    // C4[3], coeff of eps^7, polynomial in n of order 0
    -16, 97461,
    // C4[3], coeff of eps^6, polynomial in n of order 1
    -16384, 1088, 5360355,
    // C4[3], coeff of eps^5, polynomial in n of order 2
    -2560, 30464, -11560, 5360355,
    // C4[3], coeff of eps^4, polynomial in n of order 3
    35840, -34816, 17408, -3536, 1786785,
    // C4[3], coeff of eps^3, polynomial in n of order 4
    7168, -30464, 60928, -56576, 19448, 5360355,
    // C4[4], coeff of eps^7, polynomial in n of order 0
    128, 2297295,
    // C4[4], coeff of eps^6, polynomial in n of order 1
    26624, -8704, 6891885,
    // C4[4], coeff of eps^5, polynomial in n of order 2
    -77824, 34816, -6528, 6891885,
    // C4[4], coeff of eps^4, polynomial in n of order 3
    -32256, 52224, -43520, 14144, 6891885,
    // C4[5], coeff of eps^7, polynomial in n of order 0
    -6784, 8423415,
    // C4[5], coeff of eps^6, polynomial in n of order 1
    24576, -4352, 8423415,
    // C4[5], coeff of eps^5, polynomial in n of order 2
    45056, -34816, 10880, 8423415,
    // C4[6], coeff of eps^7, polynomial in n of order 0
    -1024, 3318315,
    // C4[6], coeff of eps^6, polynomial in n of order 1
    -28672, 8704, 9954945,
    // C4[7], coeff of eps^7, polynomial in n of order 0
    1024, 1640925,
  };
 #else
 #error "Bad value for GEOGRAPHICLIB_GEODESIC_ORDER"
 #endif
  int o = 0, k = 0;
  for (int l = 0; l < nC4_; ++l) {           // l is index of C4[l]
    for (int j = nC4_ - 1; j >= l; --j) {    // coeff of eps^j
      int m = nC4_ - j - 1;                  // order of polynomial in n
      _C4x[k++] = polyval(m, coeff + o, _n) / coeff[o + m + 1];
      o += m + 2;
    }
  }
  // Post condition: o == sizeof(coeff) / sizeof(double) && k == nC4x_
}

template<typename _ref = WGS84>
void Lengths(
  double eps, double sig12,
  double ssig1, double csig1, double dn1,
  double ssig2, double csig2, double dn2,
  double cbet1, double cbet2, unsigned outmask,
  double & s12b, double & m12b, double & m0,
  double & M12, double & M21,
  // Scratch area of the right size
  double Ca[])
{
  constexpr double _f1 = 1. - _ref::f;
  constexpr double _ep2 = _ref::e2 / sq(_f1);
  constexpr double _n = _ref::f / ( 2. - _ref::f);
  const double _c2 =
    (sq(_ref::a) + sq(_ref::b) *
    (_ref::e2 == 0. ? 1. :
    eatanhe(1., (_ref::f < 0 ? -1. : 1.) * sqrt(abs(_ref::e2))) / _ref::e2)) /
    2.;
  const double _etol2(0.1 * tol2_ /
    sqrt(std::max(0.001, abs(_ref::f)) * std::min(1., 1 - _ref::f / 2.) / 2.));

  // Return m12b = (reduced length)/_ref::b; also calculate s12b = distance/_ref::b,
  // and m0 = coefficient of secular term in expression for reduced length.

  outmask &= OUT_MASK;
  // outmask & DISTANCE: set s12b
  // outmask & REDUCEDLENGTH: set m12b & m0
  // outmask & GEODESICSCALE: set M12 & M21

  double m0x = 0, J12 = 0, A1 = 0, A2 = 0;
  double Cb[nC2_ + 1];
  if (outmask & (DISTANCE | REDUCEDLENGTH | GEODESICSCALE)) {
    A1 = A1m1f(eps);
    C1f(eps, Ca);
    if (outmask & (REDUCEDLENGTH | GEODESICSCALE)) {
      A2 = A2m1f(eps);
      C2f(eps, Cb);
      m0x = A1 - A2;
      A2 = 1 + A2;
    }
    A1 = 1 + A1;
  }
  if (outmask & DISTANCE) {
    double B1 = SinCosSeries(true, ssig2, csig2, Ca, nC1_) -
      SinCosSeries(true, ssig1, csig1, Ca, nC1_);
    // Missing a factor of _ref::b
    s12b = A1 * (sig12 + B1);
    if (outmask & (REDUCEDLENGTH | GEODESICSCALE)) {
      double B2 = SinCosSeries(true, ssig2, csig2, Cb, nC2_) -
        SinCosSeries(true, ssig1, csig1, Cb, nC2_);
      J12 = m0x * sig12 + (A1 * B1 - A2 * B2);
    }
  } else if (outmask & (REDUCEDLENGTH | GEODESICSCALE)) {
    // Assume here that nC1_ >= nC2_
    for (int l = 1; l <= nC2_; ++l) {
      Cb[l] = A1 * Ca[l] - A2 * Cb[l];
    }
    J12 = m0x * sig12 + (SinCosSeries(true, ssig2, csig2, Cb, nC2_) -
      SinCosSeries(true, ssig1, csig1, Cb, nC2_));
  }
  if (outmask & REDUCEDLENGTH) {
    m0 = m0x;
    // Missing a factor of _ref::b.
    // Add parens around (csig1 * ssig2) and (ssig1 * csig2) to ensure
    // accurate cancellation in the case of coincident points.
    m12b = dn2 * (csig1 * ssig2) - dn1 * (ssig1 * csig2) -
      csig1 * csig2 * J12;
  }
  if (outmask & GEODESICSCALE) {
    double csig12 = csig1 * csig2 + ssig1 * ssig2;
    double t = _ep2 * (cbet1 - cbet2) * (cbet1 + cbet2) / (dn1 + dn2);
    M12 = csig12 + (t * ssig2 - csig2 * J12) * ssig1 / dn1;
    M21 = csig12 - (t * ssig1 - csig1 * J12) * ssig2 / dn2;
  }
}


template<typename _ref = WGS84>
double Lambda12(
  double sbet1, double cbet1, double dn1,
  double sbet2, double cbet2, double dn2,
  double salp1, double calp1,
  double slam120, double clam120,
  double & salp2, double & calp2,
  double & sig12,
  double & ssig1, double & csig1,
  double & ssig2, double & csig2,
  double & eps, double & domg12,
  bool diffp, double & dlam12,
  // Scratch area of the right size
  double Ca[])
{
  constexpr double _f1 = 1. - _ref::f;
  constexpr double _ep2 = _ref::e2 / sq(_f1);
  constexpr double _n = _ref::f / ( 2. - _ref::f);
  const double _c2 =
    (sq(_ref::a) + sq(_ref::b) *
    (_ref::e2 == 0 ? 1 :
    eatanhe(1., (_ref::f < 0 ? -1 : 1) * sqrt(abs(_ref::e2))) / _ref::e2)) /
    2.;
  const double _etol2(0.1 * tol2_ /
    sqrt(std::max(0.001, abs(_ref::f)) * std::min(1., 1 - _ref::f / 2.) / 2.));

  if (sbet1 == 0 && calp1 == 0) {
    // Break degeneracy of equatorial line.  This case has already been
    // handled.
    calp1 = -tiny_;
  }

  double
  // sin(alp1) * cos(bet1) = sin(alp0)
    salp0 = salp1 * cbet1,
    calp0 = hypot(calp1, salp1 * sbet1);    // calp0 > 0

  double somg1, comg1, somg2, comg2, somg12, comg12, lam12;
  // tan(bet1) = tan(sig1) * cos(alp1)
  // tan(omg1) = sin(alp0) * tan(sig1) = tan(omg1)=tan(alp1)*sin(bet1)
  ssig1 = sbet1; somg1 = salp0 * sbet1;
  csig1 = comg1 = calp1 * cbet1;
  norm(ssig1, csig1);
  // norm(somg1, comg1); -- don't need to normalize!

  // Enforce symmetries in the case abs(bet2) = -bet1.  Need to be careful
  // about this case, since this can yield singularities in the Newton
  // iteration.
  // sin(alp2) * cos(bet2) = sin(alp0)
  salp2 = cbet2 != cbet1 ? salp0 / cbet2 : salp1;
  // calp2 = sqrt(1 - sq(salp2))
  //       = sqrt(sq(calp0) - sq(sbet2)) / cbet2
  // and subst for calp0 and rearrange to give (choose positive sqrt
  // to give alp2 in [0, pi/2]).
  calp2 = cbet2 != cbet1 || abs(sbet2) != -sbet1 ?
    sqrt(
    sq(calp1 * cbet1) +
    (cbet1 < -sbet1 ?
    (cbet2 - cbet1) * (cbet1 + cbet2) :
    (sbet1 - sbet2) * (sbet1 + sbet2))) / cbet2 :
    abs(calp1);
  // tan(bet2) = tan(sig2) * cos(alp2)
  // tan(omg2) = sin(alp0) * tan(sig2).
  ssig2 = sbet2; somg2 = salp0 * sbet2;
  csig2 = comg2 = calp2 * cbet2;
  norm(ssig2, csig2);
  // norm(somg2, comg2); -- don't need to normalize!

  // sig12 = sig2 - sig1, limit to [0, pi]
  sig12 = atan2(
    std::max(0., csig1 * ssig2 - ssig1 * csig2),
    csig1 * csig2 + ssig1 * ssig2);

  // omg12 = omg2 - omg1, limit to [0, pi]
  somg12 = std::max(0., comg1 * somg2 - somg1 * comg2);
  comg12 = comg1 * comg2 + somg1 * somg2;
  // eta = omg12 - lam120
  double eta = atan2(
    somg12 * clam120 - comg12 * slam120,
    comg12 * clam120 + somg12 * slam120);
  double B312;
  double k2 = sq(calp0) * _ep2;
  eps = k2 / (2 * (1 + sqrt(1 + k2)) + k2);
  C3f(eps, Ca);
  B312 = (SinCosSeries(true, ssig2, csig2, Ca, nC3_ - 1) -
    SinCosSeries(true, ssig1, csig1, Ca, nC3_ - 1));
  domg12 = -_ref::f * A3f(eps) * salp0 * (sig12 + B312);
  lam12 = eta + domg12;

  if (diffp) {
    if (calp2 == 0) {
      dlam12 = -2 * _f1 * dn1 / sbet1;
    } else {
      double dummy;
      Lengths<_ref>(
        eps, sig12, ssig1, csig1, dn1, ssig2, csig2, dn2,
        cbet1, cbet2, REDUCEDLENGTH,
        dummy, dlam12, dummy, dummy, dummy, Ca);
      dlam12 *= _f1 / (calp2 * cbet2);
    }
  }

  return lam12;
}

template<typename _ref = WGS84>
double InverseStart(
  double sbet1, double cbet1, double dn1,
  double sbet2, double cbet2, double dn2,
  double lam12, double slam12, double clam12,
  double & salp1, double & calp1,
  // Only updated if return val >= 0
  double & salp2, double & calp2,
  // Only updated for short lines
  double & dnm,
  // Scratch area of the right size
  double Ca[])
{
  constexpr double _f1 = 1. - _ref::f;
  constexpr double _ep2 = _ref::e2 / sq(_f1);
  constexpr double _n = _ref::f / ( 2. - _ref::f);
  const double _c2 =
    (sq(_ref::a) + sq(_ref::b) *
    (_ref::e2 == 0 ? 1 :
    eatanhe(1., (_ref::f < 0 ? -1 : 1) * sqrt(abs(_ref::e2))) / _ref::e2)) /
    2.;
  const double _etol2(0.1 * tol2_ /
    sqrt(std::max(0.001, abs(_ref::f)) * std::min(1., 1 - _ref::f / 2.) / 2.));
  // Return a starting point for Newton's method in salp1 and calp1 (function
  // value is -1).  If Newton's method doesn't need to be used, return also
  // salp2 and calp2 and function value is sig12.
  double
    sig12 = -1,                  // Return value
  // bet12 = bet2 - bet1 in [0, pi); bet12a = bet2 + bet1 in (-pi, 0]
    sbet12 = sbet2 * cbet1 - cbet2 * sbet1,
    cbet12 = cbet2 * cbet1 + sbet2 * sbet1;
  double sbet12a = sbet2 * cbet1 + cbet2 * sbet1;
  bool shortline = cbet12 >= 0 && sbet12 < 0.5 &&
    cbet2 * lam12 < 0.5;
  double somg12, comg12;
  if (shortline) {
    double sbetm2 = sq(sbet1 + sbet2);
    // sin((bet1+bet2)/2)^2
    // =  (sbet1 + sbet2)^2 / ((sbet1 + sbet2)^2 + (cbet1 + cbet2)^2)
    sbetm2 /= sbetm2 + sq(cbet1 + cbet2);
    dnm = sqrt(1 + _ep2 * sbetm2);
    double omg12 = lam12 / (_f1 * dnm);
    somg12 = sin(omg12); comg12 = cos(omg12);
  } else {
    somg12 = slam12; comg12 = clam12;
  }

  salp1 = cbet2 * somg12;
  calp1 = comg12 >= 0 ?
    sbet12 + cbet2 * sbet1 * sq(somg12) / (1 + comg12) :
    sbet12a - cbet2 * sbet1 * sq(somg12) / (1 - comg12);

  double
    ssig12 = hypot(salp1, calp1),
    csig12 = sbet1 * sbet2 + cbet1 * cbet2 * comg12;

  if (shortline && ssig12 < _etol2) {
    // really short lines
    salp2 = cbet1 * somg12;
    calp2 = sbet12 - cbet1 * sbet2 *
      (comg12 >= 0 ? sq(somg12) / (1 + comg12) : 1 - comg12);
    norm(salp2, calp2);
    // Set return value
    sig12 = atan2(ssig12, csig12);
  } else if (abs(_n) > 0.1 || csig12 >= 0. || (ssig12 >= 6. * abs(_n) * pi() * sq(cbet1))) {
    // Nothing to do, zeroth order spherical approximation is OK
  } else {
    // Scale lam12 and bet2 to x, y coordinate system where antipodal point
    // is at origin and singular point is at y = 0, x = -1.
    double y, lamscale, betscale;
    // Volatile declaration needed to fix inverse case
    // 56.320923501171 0 -56.320923501171 179.664747671772880215
    // which otherwise fails with g++ 4.4.4 x86 -O3
    volatile double x;
    double lam12x = atan2(-slam12, -clam12);    // lam12 - pi
    if (_ref::f >= 0) {               // In fact f == 0 does not get here
      // x = dlong, y = dlat
      {
        double
          k2 = sq(sbet1) * _ep2,
          eps = k2 / (2 * (1 + sqrt(1 + k2)) + k2);
        lamscale = _ref::f * cbet1 * A3f(eps) * pi();
      }
      betscale = lamscale * cbet1;

      x = lam12x / lamscale;
      y = sbet12a / betscale;
    } else {                     // _ref::f < 0
      // x = dlat, y = dlong
      double
        cbet12a = cbet2 * cbet1 - sbet2 * sbet1,
        bet12a = atan2(sbet12a, cbet12a);
      double m12b, m0, dummy;
      // In the case of lon12 = 180, this repeats a calculation made in
      // Inverse.
      Lengths<_ref>(
        _n, pi() + bet12a,
        sbet1, -cbet1, dn1, sbet2, cbet2, dn2,
        cbet1, cbet2,
        REDUCEDLENGTH, dummy, m12b, m0, dummy, dummy, Ca);
      x = -1 + m12b / (cbet1 * cbet2 * m0 * pi());
      betscale = x < -0.01 ? sbet12a / x :
        -_ref::f * sq(cbet1) * pi();
      lamscale = betscale / cbet1;
      y = lam12x / lamscale;
    }

    if (y > -tol1_ && x > -1. - xthresh_) {
      // strip near cut
      // Need double(x) here to cast away the volatility of x for min/max
      if (_ref::f >= 0) {
        salp1 = std::min(1., -x); calp1 = -sqrt(1 - sq(salp1));
      } else {
        calp1 = std::max(x > -tol1_ ? 0. : -1., x);
        salp1 = sqrt(1 - sq(calp1));
      }
    } else {
      // Estimate alp1, by solving the astroid problem.
      //
      // Could estimate alpha1 = theta + pi/2, directly, i.e.,
      //   calp1 = y/k; salp1 = -x/(1+k);  for _ref::f >= 0
      //   calp1 = x/(1+k); salp1 = -y/k;  for _ref::f < 0 (need to check)
      //
      // However, it's better to estimate omg12 from astroid and use
      // spherical formula to compute alp1.  This reduces the mean number of
      // Newton iterations for astroid cases from 2.24 (min 0, max 6) to 2.12
      // (min 0 max 5).  The changes in the number of iterations are as
      // follows:
      //
      // change percent
      //    1       5
      //    0      78
      //   -1      16
      //   -2       0.6
      //   -3       0.04
      //   -4       0.002
      //
      // The histogram of iterations is (m = number of iterations estimating
      // alp1 directly, n = number of iterations estimating via omg12, total
      // number of trials = 148605):
      //
      //  iter    m      n
      //    0   148    186
      //    1 13046  13845
      //    2 93315 102225
      //    3 36189  32341
      //    4  5396      7
      //    5   455      1
      //    6    56      0
      //
      // Because omg12 is near pi, estimate work with omg12a = pi - omg12
      double k = Astroid(x, y);
      double
        omg12a = lamscale * ( _ref::f >= 0 ? -x * k / (1 + k) : -y * (1 + k) / k );
      somg12 = sin(omg12a); comg12 = -cos(omg12a);
      // Update spherical estimate of alp1 using omg12 instead of lam12
      salp1 = cbet2 * somg12;
      calp1 = sbet12a - cbet2 * sbet1 * sq(somg12) / (1 - comg12);
    }
  }
  // Sanity check on starting guess.  Backwards check allows NaN through.
  if (!(salp1 <= 0)) {
    norm(salp1, calp1);
  } else {
    salp1 = 1; calp1 = 0;
  }
  return sig12;
}


template<typename _ref = WGS84>
double GenInverse(
  const Eigen::Ref<const Eigen::Vector2d> pt1,
  const Eigen::Ref<const Eigen::Vector2d> pt2)
{
  double lat1 = rad2deg(pt1.x());
  double lon1 = rad2deg(pt1.y());

  double lat2 = rad2deg(pt2.x());
  double lon2 = rad2deg(pt2.y());

  constexpr double _f1 = 1. - _ref::f;
  constexpr double _ep2 = _ref::e2 / sq(_f1);
  constexpr double _n = _ref::f / ( 2. - _ref::f);
  const double _c2 =
    (sq(_ref::a) + sq(_ref::b) *
    (_ref::e2 == 0 ? 1 :
    eatanhe(1., (_ref::f < 0 ? -1 : 1) * sqrt(abs(_ref::e2))) / _ref::e2)) /
    2.;
  const double _etol2(0.1 * tol2_ /
    sqrt(std::max(0.001, abs(_ref::f)) * std::min(1., 1 - _ref::f / 2.) / 2.));

  double s12;
  double salp1;
  double calp1;
  double salp2;
  double calp2;
  double m12;
  double M12;
  double M21;
  double S12;

  unsigned outmask = DISTANCE;

  // Compute longitude difference (AngDiff does this carefully).  Result is
  // in [-180, 180] but -180 is only for west-going geodesics.  180 is for
  // east-going and meridional geodesics.
  double lon12s, lon12 = AngDiff(lon1, lon2, lon12s);
  // Make longitude difference positive.
  int lonsign = lon12 >= 0 ? 1 : -1;
  // If very close to being on the same half-meridian, then make it so.
  lon12 = lonsign * AngRound(lon12);
  lon12s = AngRound((180 - lon12) - lonsign * lon12s);
  double lam12 = lon12 * M_PI / 180.;
  double slam12;
  double clam12;
  if (lon12 > 90) {
    sincosd(lon12s, slam12, clam12);
    clam12 = -clam12;
  } else {
    sincosd(lon12, slam12, clam12);
  }

  // If really close to the equator, treat as on equator.
  lat1 = AngRound(LatFix(lat1));
  lat2 = AngRound(LatFix(lat2));
  // Swap points so that point with higher (abs) latitude is point 1.
  // If one latitude is a nan, then it becomes lat1.
  int swapp = abs(lat1) < abs(lat2) ? -1 : 1;
  if (swapp < 0) {
    lonsign *= -1;
    std::swap(lat1, lat2);
  }
  // Make lat1 <= 0
  int latsign = lat1 < 0 ? 1 : -1;
  lat1 *= latsign;
  lat2 *= latsign;
  // Now we have
  //
  //     0 <= lon12 <= 180
  //     -90 <= lat1 <= 0
  //     lat1 <= lat2 <= -lat1
  //
  // longsign, swapp, latsign register the transformation to bring the
  // coordinates to this canonical form.  In all cases, 1 means no change was
  // made.  We make these transformations so that there are few cases to
  // check, e.g., on verifying quadrants in atan2.  In addition, this
  // enforces some symmetries in the results returned.

  double sbet1, cbet1, sbet2, cbet2, s12x, m12x;

  sincosd(lat1, sbet1, cbet1); sbet1 *= _f1;
  // Ensure cbet1 = +epsilon at poles; doing the fix on beta means that sig12
  // will be <= 2*tiny for two points at the same pole.
  norm(sbet1, cbet1); cbet1 = std::max(tiny_, cbet1);

  sincosd(lat2, sbet2, cbet2); sbet2 *= _f1;
  // Ensure cbet2 = +epsilon at poles
  norm(sbet2, cbet2); cbet2 = std::max(tiny_, cbet2);

  // If cbet1 < -sbet1, then cbet2 - cbet1 is a sensitive measure of the
  // |bet1| - |bet2|.  Alternatively (cbet1 >= -sbet1), abs(sbet2) + sbet1 is
  // a better measure.  This logic is used in assigning calp2 in Lambda12.
  // Sometimes these quantities vanish and in that case we force bet2 = +/-
  // bet1 exactly.  An example where is is necessary is the inverse problem
  // 48.522876735459 0 -48.52287673545898293 179.599720456223079643
  // which failed with Visual Studio 10 (Release and Debug)

  if (cbet1 < -sbet1) {
    if (cbet2 == cbet1) {
      sbet2 = sbet2 < 0 ? sbet1 : -sbet1;
    }
  } else {
    if (abs(sbet2) == -sbet1) {
      cbet2 = cbet1;
    }
  }

  double
    dn1 = sqrt(1 + _ep2 * sq(sbet1)),
    dn2 = sqrt(1 + _ep2 * sq(sbet2));

  double a12, sig12;
  // index zero element of this array is unused
  double Ca[nC_];

  bool meridian = lat1 == -90 || slam12 == 0;

  if (meridian) {
    // Endpoints are on a single full meridian, so the geodesic might lie on
    // a meridian.

    calp1 = clam12; salp1 = slam12;    // Head to the target longitude
    calp2 = 1; salp2 = 0;              // At the target we're heading north

    double
    // tan(bet) = tan(sig) * cos(alp)
      ssig1 = sbet1, csig1 = calp1 * cbet1,
      ssig2 = sbet2, csig2 = calp2 * cbet2;

    // sig12 = sig2 - sig1
    sig12 = atan2(
      std::max(0., csig1 * ssig2 - ssig1 * csig2),
      csig1 * csig2 + ssig1 * ssig2);
    {
      double dummy;
      Lengths<_ref>(
        _n, sig12, ssig1, csig1, dn1, ssig2, csig2, dn2, cbet1, cbet2,
        DISTANCE | REDUCEDLENGTH,
        s12x, m12x, dummy, M12, M21, Ca);
    }
    // Add the check for sig12 since zero length geodesics might yield m12 <
    // 0.  Test case was
    //
    //    echo 20.001 0 20.001 0 | GeodSolve -i
    //
    // In fact, we will have sig12 > pi/2 for meridional geodesic which is
    // not a shortest path.
    if (sig12 < 1 || m12x >= 0) {
      // Need at least 2, to handle 90 0 90 180
      if (sig12 < 3 * tiny_) {
        sig12 = m12x = s12x = 0;
      }
      m12x *= _ref::b;
      s12x *= _ref::b;
      a12 = sig12 / M_PI / 180.;
    } else {
      // m12 < 0, i.e., prolate and too close to anti-podal
      meridian = false;
    }
  }

  // somg12 > 1 marks that it needs to be calculated
  double omg12 = 0, somg12 = 2, comg12 = 0;
  if (!meridian &&
    sbet1 == 0 &&        // and sbet2 == 0
    (_ref::f <= 0 || lon12s >= _ref::f * 180))
  {
    // Geodesic runs along equator
    calp1 = calp2 = 0; salp1 = salp2 = 1;
    s12x = _ref::a * lam12;
    sig12 = omg12 = lam12 / _f1;
    m12x = _ref::b * sin(sig12);
    if (outmask & GEODESICSCALE) {
      M12 = M21 = cos(sig12);
    }
    a12 = lon12 / _f1;

  } else if (!meridian) {
    // Now point1 and point2 belong within a hemisphere bounded by a
    // meridian and geodesic is neither meridional or equatorial.

    // Figure a starting point for Newton's method
    double dnm;
    sig12 = InverseStart<_ref>(
      sbet1, cbet1, dn1, sbet2, cbet2, dn2,
      lam12, slam12, clam12,
      salp1, calp1, salp2, calp2, dnm,
      Ca);

    if (sig12 >= 0) {
      // Short lines (InverseStart sets salp2, calp2, dnm)
      s12x = sig12 * _ref::b * dnm;
      m12x = sq(dnm) * _ref::b * sin(sig12 / dnm);
      if (outmask & GEODESICSCALE) {
        M12 = M21 = cos(sig12 / dnm);
      }
      a12 = sig12 / M_PI / 180.;
      omg12 = lam12 / (_f1 * dnm);
    } else {
      // Newton's method.  This is a straightforward solution of f(alp1) =
      // lambda12(alp1) - lam12 = 0 with one wrinkle.  f(alp) has exactly one
      // root in the interval (0, pi) and its derivative is positive at the
      // root.  Thus f(alp) is positive for alp > alp1 and negative for alp <
      // alp1.  During the course of the iteration, a range (alp1a, alp1b) is
      // maintained which brackets the root and with each evaluation of
      // f(alp) the range is shrunk, if possible.  Newton's method is
      // restarted whenever the derivative of f is negative (because the new
      // value of alp1 is then further from the solution) or if the new
      // estimate of alp1 lies outside (0,pi); in this case, the new starting
      // guess is taken to be (alp1a + alp1b) / 2.
      //
      // initial values to suppress warnings (if loop is executed 0 times)
      double ssig1 = 0, csig1 = 0, ssig2 = 0, csig2 = 0, eps = 0, domg12 = 0;
      unsigned numit = 0;
      // Bracketing range
      double salp1a = tiny_, calp1a = 1, salp1b = tiny_, calp1b = -1;
      for (bool tripn = false, tripb = false;
        numit < maxit2_;
        ++numit)
      {
        // the WGS84 test set: mean = 1.47, sd = 1.25, max = 16
        // WGS84 and random input: mean = 2.85, sd = 0.60
        double dv;
        double v = Lambda12<_ref>(
          sbet1, cbet1, dn1, sbet2, cbet2, dn2, salp1, calp1,
          slam12, clam12,
          salp2, calp2, sig12, ssig1, csig1, ssig2, csig2,
          eps, domg12, numit < maxit1_, dv, Ca);
        // Reversed test to allow escape with NaNs
        if (tripb || !(abs(v) >= (tripn ? 8 : 1) * tol0_)) {break;}
        // Update bracketing values
        if (v > 0 && (numit > maxit1_ || calp1 / salp1 > calp1b / salp1b)) {
          salp1b = salp1; calp1b = calp1;
        } else if (v < 0 && (numit > maxit1_ || calp1 / salp1 < calp1a / salp1a)) {
          salp1a = salp1; calp1a = calp1;
        }
        if (numit < maxit1_ && dv > 0) {
          double
            dalp1 = -v / dv;
          double
            sdalp1 = sin(dalp1), cdalp1 = cos(dalp1),
            nsalp1 = salp1 * cdalp1 + calp1 * sdalp1;
          if (nsalp1 > 0 && abs(dalp1) < pi()) {
            calp1 = calp1 * cdalp1 - salp1 * sdalp1;
            salp1 = nsalp1;
            norm(salp1, calp1);
            // In some regimes we don't get quadratic convergence because
            // slope -> 0.  So use convergence conditions based on epsilon
            // instead of sqrt(epsilon).
            tripn = abs(v) <= 16 * tol0_;
            continue;
          }
        }
        // Either dv was not positive or updated value was outside legal
        // range.  Use the midpoint of the bracket as the next estimate.
        // This mechanism is not needed for the WGS84 ellipsoid, but it does
        // catch problems with more eccentric ellipsoids.  Its efficacy is
        // such for the WGS84 test set with the starting guess set to alp1 =
        // 90deg:
        // the WGS84 test set: mean = 5.21, sd = 3.93, max = 24
        // WGS84 and random input: mean = 4.74, sd = 0.99
        salp1 = (salp1a + salp1b) / 2;
        calp1 = (calp1a + calp1b) / 2;
        norm(salp1, calp1);
        tripn = false;
        tripb = (abs(salp1a - salp1) + (calp1a - calp1) < tolb_ ||
          abs(salp1 - salp1b) + (calp1 - calp1b) < tolb_);
      }
      {
        double dummy;
        // Ensure that the reduced length and geodesic scale are computed in
        // a "canonical" way, with the I2 integral.
        unsigned lengthmask = outmask |
          (outmask & (REDUCEDLENGTH | GEODESICSCALE) ? DISTANCE : NONE);
        Lengths<_ref>(
          eps, sig12, ssig1, csig1, dn1, ssig2, csig2, dn2,
          cbet1, cbet2, lengthmask, s12x, m12x, dummy, M12, M21, Ca);
      }
      m12x *= _ref::b;
      s12x *= _ref::b;
      a12 = sig12 / M_PI / 180.;
      if (outmask & AREA) {
        // omg12 = lam12 - domg12
        double sdomg12 = sin(domg12), cdomg12 = cos(domg12);
        somg12 = slam12 * cdomg12 - clam12 * sdomg12;
        comg12 = clam12 * cdomg12 + slam12 * sdomg12;
      }
    }
  }

  if (outmask & DISTANCE) {
    s12 = 0 + s12x;              // Convert -0 to 0
  }
  if (outmask & REDUCEDLENGTH) {
    m12 = 0 + m12x;              // Convert -0 to 0
  }
  if (outmask & AREA) {
    double
    // From Lambda12: sin(alp1) * cos(bet1) = sin(alp0)
      salp0 = salp1 * cbet1,
      calp0 = hypot(calp1, salp1 * sbet1);    // calp0 > 0
    double alp12;
    if (calp0 != 0 && salp0 != 0) {
      double
      // From Lambda12: tan(bet) = tan(sig) * cos(alp)
        ssig1 = sbet1, csig1 = calp1 * cbet1,
        ssig2 = sbet2, csig2 = calp2 * cbet2,
        k2 = sq(calp0) * _ep2,
        eps = k2 / (2 * (1 + sqrt(1 + k2)) + k2),
      // Multiplier = a^2 * e^2 * cos(alpha0) * sin(alpha0).
        A4 = sq(_ref::a) * calp0 * salp0 * _ref::e2;
      norm(ssig1, csig1);
      norm(ssig2, csig2);
      C4f(eps, Ca);
      double
        B41 = SinCosSeries(false, ssig1, csig1, Ca, nC4_),
        B42 = SinCosSeries(false, ssig2, csig2, Ca, nC4_);
      S12 = A4 * (B42 - B41);
    } else {
      // Avoid problems with indeterminate sig1, sig2 on equator
      S12 = 0;
    }

    if (!meridian && somg12 > 1) {
      somg12 = sin(omg12); comg12 = cos(omg12);
    }

    if (!meridian &&
      // omg12 < 3/4 * pi
      comg12 > -0.7071 &&          // Long difference not too big
      sbet2 - sbet1 < 1.75)        // Lat difference not too big
    {    // Use tan(Gamma/2) = tan(omg12/2)
         // * (tan(bet1/2)+tan(bet2/2))/(1+tan(bet1/2)*tan(bet2/2))
         // with tan(x/2) = sin(x)/(1+cos(x))
      double domg12 = 1 + comg12, dbet1 = 1 + cbet1, dbet2 = 1 + cbet2;
      alp12 = 2 * atan2(
        somg12 * ( sbet1 * dbet2 + sbet2 * dbet1 ),
        domg12 * ( sbet1 * sbet2 + dbet1 * dbet2 ) );
    } else {
      // alp12 = alp2 - alp1, used in atan2 so no need to normalize
      double
        salp12 = salp2 * calp1 - calp2 * salp1,
        calp12 = calp2 * calp1 + salp2 * salp1;
      // The right thing appears to happen if alp1 = +/-180 and alp2 = 0, viz
      // salp12 = -0 and alp12 = -180.  However this depends on the sign
      // being attached to 0 correctly.  The following ensures the correct
      // behavior.
      if (salp12 == 0 && calp12 < 0) {
        salp12 = tiny_ * calp1;
        calp12 = -1;
      }
      alp12 = atan2(salp12, calp12);
    }
    S12 += _c2 * alp12;
    S12 *= swapp * lonsign * latsign;
    // Convert -0 to 0
    S12 += 0;
  }

  // Convert calp, salp to azimuth accounting for lonsign, swapp, latsign.
  if (swapp < 0) {
    std::swap(salp1, salp2);
    std::swap(calp1, calp2);
    if (outmask & GEODESICSCALE) {
      std::swap(M12, M21);
    }
  }

  salp1 *= swapp * lonsign; calp1 *= swapp * latsign;
  salp2 *= swapp * lonsign; calp2 *= swapp * latsign;

  // Returned value in [0, 180]
  return s12;
}

}  // namespace cbr::geo

#endif  // CBR_MATH__GEODESIC_HPP_
