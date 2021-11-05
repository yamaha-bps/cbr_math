// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_MATH__GEODETIC_HPP_
#define CBR_MATH__GEODETIC_HPP_

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <exception>
#include <utility>

#include "geodetic_data.hpp"
#include "math.hpp"

namespace cbr::geo
{

/***************************************************************************
 * \brief Converts geographic coordinates into geocentric ones.
 ***************************************************************************/
template<typename _ref = WGS84>
void geographic2geocentric(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  Eigen::Ref<Eigen::Vector3d> llr)
{
  constexpr double b2a2 = _ref::b2 / _ref::a2;
  const double theta = atan(tan(lla[0]) * b2a2);
  const double cosTheta = cos(theta);
  const double sinTheta = sin(theta);

  const double r =
    1. / sqrt((cosTheta * cosTheta) / (_ref::a2) +(sinTheta * sinTheta) / (_ref::b2));

  const double thetaPrime =
    atan2(r * sinTheta + lla[2] * sin(lla[0]), r * cosTheta + lla[2] * cos(lla[0]));

  const double rPrime = (r * cosTheta + lla[2] * cos(lla[0])) / cos(thetaPrime);

  llr[0] = thetaPrime;
  llr[1] = lla[1];
  llr[2] = rPrime;
}

template<typename _ref = WGS84>
Eigen::Vector3d geographic2geocentric(const Eigen::Ref<const Eigen::Vector3d> lla)
{
  Eigen::Vector3d llr;
  geographic2geocentric<_ref>(lla, llr);
  return llr;
}


/***************************************************************************
 * \brief Converts geocentric coordinates into earth-centered-earth-fixed.
 ***************************************************************************/
template<typename _ref = WGS84>
void llr2ecef(
  const Eigen::Ref<const Eigen::Vector3d> llr,
  Eigen::Ref<Eigen::Vector3d> ecef)
{
  const double cosTheta = cos(llr[0]);
  const double sinTheta = sin(llr[0]);

  ecef[0] = llr[2] * cosTheta * cos(llr[1]);
  ecef[1] = llr[2] * cosTheta * sin(llr[1]);
  ecef[2] = llr[2] * sinTheta;
}

template<typename _ref = WGS84>
Eigen::Vector3d llr2ecef(const Eigen::Ref<const Eigen::Vector3d> llr)
{
  Eigen::Vector3d ecef;
  llr2ecef<_ref>(llr, ecef);
  return ecef;
}


/***************************************************************************
 * \brief Converts geographic coordinates into earth-centered-earth-fixed.
 ***************************************************************************/
template<typename _ref = WGS84>
void lla2ecef(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  Eigen::Ref<Eigen::Vector3d> ecef)
{
  geographic2geocentric<_ref>(lla, ecef);
  llr2ecef<_ref>(ecef, ecef);
}

template<typename _ref = WGS84>
Eigen::Vector3d lla2ecef(const Eigen::Ref<const Eigen::Vector3d> lla)
{
  Eigen::Vector3d ecef;
  lla2ecef<_ref>(lla, ecef);
  return ecef;
}

// forward-declaration
template<typename _ref = WGS84>
Eigen::Matrix3d geo2ecef(const Eigen::Ref<const Eigen::Vector2d> ll);

template<typename _ref = WGS84, typename derived>
Eigen::Isometry3d lla2ecef(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  const Eigen::QuaternionBase<derived> & R_NWU)
{
  const Eigen::Vector3d T_W_S = lla2ecef<_ref>(lla);
  const Eigen::Quaterniond R_W_NWU(geo2ecef<_ref>(lla.template head<2>()));

  return Eigen::Translation3d(T_W_S) * (R_W_NWU * R_NWU);
}

/***************************************************************************
 * \brief Converts earth-centered-earth-fixed (ecef) coordinates into
 * geographic latitude, longitude and altitude (lla)
 ***************************************************************************/
template<typename _ref = WGS84>
void ecef2lla(
  const Eigen::Ref<const Eigen::Vector3d> ecef,
  Eigen::Ref<Eigen::Vector3d> lla)
{
  const double x = ecef[0];
  const double y = ecef[1];
  const double z = ecef[2];

  double & lat = lla[0];
  double & lon = lla[1];
  double & ht = lla[2];

  constexpr double a1 = _ref::a * _ref::e2;
  constexpr double a2 = a1 * a1;
  constexpr double a3 = a1 * _ref::e2 / 2.;
  constexpr double a4 = (5. / 2.) * a2;
  constexpr double a5 = a1 + a3;
  constexpr double a6 = 1. - _ref::e2;

  double zp, w2, w, z2, r2, r, s2, c2, s, c, ss;  // NOLINT
  double g, rg, rf, u, v, m, f, p;  // NOLINT
  zp = fabs(z);
  w2 = x * x + y * y;
  w = sqrt(w2);
  z2 = z * z;
  r2 = w2 + z2;
  r = sqrt(r2);
  if (r < 100000.) {
    lat = 0.;
    lon = 0.;
    ht = -1.e7;
    return;
  }

  lon = atan2(y, x);
  s2 = z2 / r2;
  c2 = w2 / r2;
  u = a2 / r;
  v = a3 - a4 / r;

  if (c2 > .3) {
    s = (zp / r) * (1. + c2 * (a1 + u + s2 * v) / r);
    lat = asin(s);
    ss = s * s;
    c = sqrt(1. - ss);
  } else {
    c = (w / r) * (1. - s2 * (a5 - u - c2 * v) / r);
    lat = acos(c);
    ss = 1. - c * c;
    s = sqrt(ss);
  }

  g = 1. - _ref::e2 * ss;
  rg = _ref::a / sqrt(g);
  rf = a6 * rg;
  u = w - rg * c;
  v = zp - rf * s;
  f = c * u + s * v;
  m = c * v - s * u;
  p = m / (rf / g + f);
  lat = lat + p;
  ht = f + m * p / 2.;

  if (z < 0.) {
    lat = -lat;
  }
}

template<typename _ref = WGS84>
Eigen::Vector3d ecef2lla(const Eigen::Ref<const Eigen::Vector3d> ecef)
{
  Eigen::Vector3d lla;
  ecef2lla<_ref>(ecef, lla);
  return lla;
}

template<typename _ref = WGS84>
std::pair<Eigen::Vector3d, Eigen::Quaterniond> ecef2lla(const Eigen::Isometry3d & P_ecef)
{
  const Eigen::Vector3d lla = ecef2lla<_ref>(P_ecef.translation());
  const Eigen::Matrix3d R_W_NWU = geo2ecef<_ref>(lla.head<2>());

  return {lla, Eigen::Quaterniond(R_W_NWU.transpose() * P_ecef.rotation())};
}


/***************************************************************************
 * \brief Converts earth-centered-earth-fixed (ecef) coordinates into
 * geocentric latitude, longitude and radius (llr)
 ***************************************************************************/
template<typename _ref = WGS84>
void ecef2llr(
  const Eigen::Ref<const Eigen::Vector3d> ecef,
  Eigen::Ref<Eigen::Vector3d> llr)
{
  const double z = ecef[2];
  llr[2] = ecef.norm();
  llr[1] = atan2(ecef[1], ecef[0]);
  llr[0] = asin(z / llr[2]);
}

template<typename _ref = WGS84>
Eigen::Vector3d ecef2llr(const Eigen::Ref<const Eigen::Vector3d> ecef)
{
  Eigen::Vector3d llr;
  ecef2llr<_ref>(ecef, llr);
  return llr;
}


/***************************************************************************
 * \brief Converts north-west-up geographic or geocentric orientation into
 * earth-centered-earth-fixed one. Latitude and longitude must be in the
 * same frame as the orientation.
 ***************************************************************************/
template<typename _ref = WGS84>
void geo2ecef(
  const Eigen::Ref<const Eigen::Vector2d> ll,
  Eigen::Ref<Eigen::Matrix3d> M_ecef_ll)
{
  const double cosTheta = cos(ll[0]);
  const double sinTheta = sin(ll[0]);
  const double cosLong = cos(ll[1]);
  const double sinLong = sin(ll[1]);

  M_ecef_ll <<
    -sinTheta * cosLong, sinLong, cosTheta * cosLong,
    -sinTheta * sinLong, -cosLong, cosTheta * sinLong,
    cosTheta, 0., sinTheta;
}

template<typename _ref>
Eigen::Matrix3d geo2ecef(
  const Eigen::Ref<const Eigen::Vector2d> ll)
{
  Eigen::Matrix3d M_ecef_ll;
  geo2ecef<_ref>(ll, M_ecef_ll);
  return M_ecef_ll;
}

template<typename _ref = WGS84,
  typename derivedIn,
  typename derivedOut>
void geo2ecef(
  const Eigen::QuaternionBase<derivedIn> & qIn,
  const Eigen::Ref<const Eigen::Vector2d> ll,
  Eigen::QuaternionBase<derivedOut> & qOut)
{
  qOut = Eigen::Quaterniond(geo2ecef<_ref>(ll)) * qIn;
}

template<typename _ref = WGS84, typename derived>
Eigen::Quaterniond geo2ecef(
  const Eigen::QuaternionBase<derived> & qIn,
  const Eigen::Ref<const Eigen::Vector2d> ll)
{
  Eigen::Quaterniond qOut;
  geo2ecef<_ref>(qIn, ll, qOut);
  return qOut;
}

/***************************************************************************
 * \brief Converts geocentric north-west-up orientation into geographic one
 *  given geographic coordinates.
 ***************************************************************************/
template<typename _ref = WGS84,
  typename derivedIn,
  typename derivedOut>
void imu2nwu(
  const Eigen::QuaternionBase<derivedIn> & qIn,
  const Eigen::Ref<const Eigen::Vector3d> lla,
  Eigen::QuaternionBase<derivedOut> & qOut)
{
  const Eigen::Vector3d llr = geographic2geocentric<_ref>(lla);
  const Eigen::Matrix3d qEC = geo2ecef<_ref>(llr.head<2>());
  const Eigen::Matrix3d qEG = geo2ecef<_ref>(lla.head<2>());

  qOut = (qEG.transpose() * qEC) * qIn;
}

template<typename _ref = WGS84, typename derived>
Eigen::Quaterniond imu2nwu(
  const Eigen::QuaternionBase<derived> & qIn,
  const Eigen::Ref<const Eigen::Vector3d> lla)
{
  Eigen::Quaterniond qOut;
  imu2nwu<_ref>(qIn, lla, qOut);
  return qOut;
}

/***************************************************************************
 * \brief Converts geographic north-west-up orientation into geocentric one
 *  given geographic coordinates.
 ***************************************************************************/
template<typename _ref = WGS84,
  typename derivedIn,
  typename derivedOut>
void nwu2imu(
  const Eigen::QuaternionBase<derivedIn> & qIn,
  const Eigen::Ref<const Eigen::Vector3d> lla,
  Eigen::QuaternionBase<derivedOut> & qOut)
{
  const Eigen::Vector3d llr = geographic2geocentric<_ref>(lla);
  const Eigen::Matrix3d qEC = geo2ecef<_ref>(llr.head<2>());
  const Eigen::Matrix3d qEG = geo2ecef<_ref>(lla.head<2>());

  qOut = (qEC.transpose() * qEG) * qIn;
}

template<typename _ref = WGS84, typename derived>
Eigen::Quaterniond nwu2imu(
  const Eigen::QuaternionBase<derived> & qIn,
  const Eigen::Ref<const Eigen::Vector3d> lla)
{
  Eigen::Quaterniond qOut;
  nwu2imu<_ref>(qIn, lla, qOut);
  return qOut;
}

/***************************************************************************
 * \brief Computes gnomonic projection of lla onto the plane tangent to the
 * ellipsoid at llaRef (the altitude of llaRef is ignored). The projection
 * is done from a point with a geographic altitude equal to 0, and the z
 * component of the result is the original geographic altitude. If the
 * projection is impossible, the function returns false, otherwise it
 * returns true.
 ***************************************************************************/
template<typename _ref = WGS84>
bool lla2gnomonic(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  Eigen::Ref<Eigen::Vector3d> xyz)
{
  const double cosLatRef = cos(llaRef[0]);
  const double sinLatRef = sin(llaRef[0]);
  const double cosLongRef = cos(llaRef[1]);
  const double sinLongRef = sin(llaRef[1]);

  const Eigen::Vector3d x{
    -sinLatRef * cosLongRef,
    -sinLatRef * sinLongRef,
    cosLatRef};

  const Eigen::Vector3d y{
    sinLongRef,
    -cosLongRef,
    0.};

  const Eigen::Vector3d z{
    cosLatRef * cosLongRef,
    cosLatRef * sinLongRef,
    sinLatRef};

  const Eigen::Vector3d ecef = lla2ecef<_ref>(Eigen::Vector3d(lla[0], lla[1], 0.));
  if (z.dot(ecef) < 1e-9) {
    return false;
  }

  const Eigen::Vector3d ecefRef = lla2ecef<_ref>(Eigen::Vector3d(llaRef[0], llaRef[1], 0.));

  Eigen::Matrix3d M;
  M.col(0) = -x;
  M.col(1) = -y;
  M.col(2) = ecef;

  const double alti = lla[2];

  xyz = M.fullPivLu().solve(ecefRef);

  if (xyz[2] <= 0.) {
    return false;
  }

  xyz[2] = alti;

  return true;
}

template<typename _ref = WGS84>
std::pair<bool, Eigen::Vector3d> lla2gnomonic(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  const Eigen::Ref<const Eigen::Vector3d> llaRef)
{
  std::pair<bool, Eigen::Vector3d> out;
  out.first = lla2gnomonic<_ref>(lla, llaRef, out.second);
  return out;
}


/***************************************************************************
 * \brief Computes inverse gnomonic projection from the plane tangent to the
 * ellipsoid at llaRef (the altitude of llaRef is ignored). The altitude of
 * the result is the z component of the projection.
 ***************************************************************************/
template<typename _ref = WGS84>
void gnomonic2lla(
  const Eigen::Ref<const Eigen::Vector3d> xyz,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  Eigen::Ref<Eigen::Vector3d> lla)
{
  const double cosLatRef = cos(llaRef[0]);
  const double sinLatRef = sin(llaRef[0]);
  const double cosLongRef = cos(llaRef[1]);
  const double sinLongRef = sin(llaRef[1]);

  const Eigen::Vector3d x{
    -sinLatRef * cosLongRef,
    -sinLatRef * sinLongRef,
    cosLatRef};

  const Eigen::Vector3d y{
    sinLongRef,
    -cosLongRef,
    0.};

  Eigen::Vector3d ecef = lla2ecef<_ref>(Eigen::Vector3d(llaRef[0], llaRef[1], 0.));

  ecef += xyz[0] * x + xyz[1] * y;

  const double alti = xyz[2];

  ecef2llr<_ref>(ecef, lla);
  constexpr double a2b2 = _ref::a2 / _ref::b2;
  lla[0] = atan(a2b2 * tan(lla[0]));
  lla[2] = alti;
}

template<typename _ref = WGS84>
Eigen::Vector3d gnomonic2lla(
  const Eigen::Ref<const Eigen::Vector3d> xyz,
  const Eigen::Ref<const Eigen::Vector3d> llaRef)
{
  Eigen::Vector3d lla;
  gnomonic2lla<_ref>(xyz, llaRef, lla);
  return lla;
}


/***************************************************************************
 * \brief Converts geographic coordinates into north-west-up ones.
 ***************************************************************************/
template<typename _ref = WGS84>
void lla2nwu(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  Eigen::Ref<Eigen::Vector3d> nwu)
{
  const Eigen::Vector3d ecefRef = lla2ecef<_ref>(llaRef);
  const Eigen::Vector3d ecef = lla2ecef<_ref>(lla);
  nwu = geo2ecef<_ref>(llaRef.head<2>()).transpose() * (ecef - ecefRef);
}

template<typename _ref = WGS84>
Eigen::Vector3d lla2nwu(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  const Eigen::Ref<const Eigen::Vector3d> llaRef)
{
  Eigen::Vector3d nwu;
  lla2nwu<_ref>(lla, llaRef, nwu);
  return nwu;
}

template<typename _ref = WGS84,
  typename derived>
void lla2nwu(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  const Eigen::QuaternionBase<derived> & qRef,
  Eigen::Ref<Eigen::Vector3d> nwu)
{
  lla2nwu<_ref>(lla, llaRef, nwu);
  nwu = qRef.toRotationMatrix().transpose() * nwu;
}

template<typename _ref = WGS84,
  typename derived>
Eigen::Vector3d lla2nwu(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  const Eigen::QuaternionBase<derived> & qRef)
{
  Eigen::Vector3d nwu;
  lla2nwu<_ref>(lla, llaRef, qRef, nwu);
  return nwu;
}

/***************************************************************************
 * \brief Transforms a geographic frame into north-west-up one.
 ***************************************************************************/
template<typename _ref = WGS84,
  typename derived1,
  typename derived2,
  typename derived3>
void lla2nwu(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  const Eigen::QuaternionBase<derived1> & q,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  const Eigen::QuaternionBase<derived2> & qRef,
  Eigen::Ref<Eigen::Vector3d> nwu,
  Eigen::QuaternionBase<derived3> & qNwu)
{
  const Eigen::Vector3d ecefFrame = lla2ecef<_ref>(lla);
  const Eigen::Vector3d ecefRef = lla2ecef<_ref>(llaRef);

  const Eigen::Quaterniond qEcefFrame = geo2ecef<_ref>(q, lla.head<2>());
  const Eigen::Quaterniond qEcefRefInv = geo2ecef<_ref>(qRef, llaRef.head<2>()).conjugate();

  nwu = qEcefRefInv.toRotationMatrix() * (ecefFrame - ecefRef);
  qNwu = qEcefRefInv * qEcefFrame;
}

template<typename _ref = WGS84,
  typename derived1,
  typename derived2>
void lla2nwu(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  const Eigen::QuaternionBase<derived1> & q,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  Eigen::Ref<Eigen::Vector3d> nwu,
  Eigen::QuaternionBase<derived2> & qNwu)
{
  lla2nwu<_ref>(lla, q, llaRef, Eigen::Quaterniond::Identity(), nwu, qNwu);
}

/***************************************************************************
 * \brief Converts north-west-up coordinates into geographic ones.
 ***************************************************************************/
template<typename _ref = WGS84>
void nwu2lla(
  const Eigen::Ref<const Eigen::Vector3d> nwu,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  Eigen::Ref<Eigen::Vector3d> lla)
{
  const Eigen::Vector3d ecefRef = lla2ecef<_ref>(llaRef);
  lla = ecefRef + geo2ecef<_ref>(llaRef.head<2>()) * nwu;
  ecef2lla<_ref>(lla, lla);
}

template<typename _ref = WGS84>
Eigen::Vector3d nwu2lla(
  const Eigen::Ref<const Eigen::Vector3d> nwu,
  const Eigen::Ref<const Eigen::Vector3d> llaRef)
{
  Eigen::Vector3d lla;
  nwu2lla<_ref>(nwu, llaRef, lla);
  return lla;
}

template<typename _ref = WGS84,
  typename derived>
void nwu2lla(
  const Eigen::Ref<const Eigen::Vector3d> nwu,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  const Eigen::QuaternionBase<derived> & qRef,
  Eigen::Ref<Eigen::Vector3d> lla)
{
  lla = qRef.toRotationMatrix() * nwu;
  nwu2lla<_ref>(lla, llaRef, lla);
}

template<typename _ref = WGS84,
  typename derived>
Eigen::Vector3d nwu2lla(
  const Eigen::Ref<const Eigen::Vector3d> nwu,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  const Eigen::QuaternionBase<derived> & qRef)
{
  Eigen::Vector3d lla;
  nwu2lla<_ref>(nwu, llaRef, qRef, lla);
  return lla;
}

/***************************************************************************
 * \brief Transforms a north-west-up frame into a geographic one.
 ***************************************************************************/
template<typename _ref = WGS84,
  typename derived1,
  typename derived2,
  typename derived3>
void nwu2lla(
  const Eigen::Ref<const Eigen::Vector3d> nwu,
  const Eigen::QuaternionBase<derived1> & q,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  const Eigen::QuaternionBase<derived2> & qRef,
  Eigen::Ref<Eigen::Vector3d> lla,
  Eigen::QuaternionBase<derived3> & qLla)
{
  const Eigen::Vector3d ecefRef = lla2ecef<_ref>(llaRef);
  const Eigen::Quaterniond qEcefRef = geo2ecef<_ref>(qRef, llaRef.head<2>());
  const Eigen::Quaterniond qEcefFrame = qEcefRef * q;

  const Eigen::Vector3d ecefFrame = ecefRef + qEcefRef.toRotationMatrix() * nwu;
  ecef2lla<_ref>(ecefFrame, lla);

  const Eigen::Matrix3d RotEcefFrameInv = geo2ecef<_ref>(lla.head<2>()).transpose();
  qLla = RotEcefFrameInv * qEcefFrame;
}

template<typename _ref = WGS84,
  typename derived1,
  typename derived2>
void nwu2lla(
  const Eigen::Ref<const Eigen::Vector3d> lla,
  const Eigen::QuaternionBase<derived1> & q,
  const Eigen::Ref<const Eigen::Vector3d> llaRef,
  Eigen::Ref<Eigen::Vector3d> nwu,
  Eigen::QuaternionBase<derived2> & qNwu)
{
  nwu2lla<_ref>(lla, q, llaRef, Eigen::Quaterniond::Identity(), nwu, qNwu);
}

/***************************************************************************
 * \brief Returns cartesian distance in the earth-centered-earth-fixed frame
 * for 2 points given by their geographic latitude, longitude and altitude
 ***************************************************************************/
template<typename _ref = WGS84>
double distCartesian(
  const Eigen::Ref<const Eigen::Vector3d> pt1,
  const Eigen::Ref<const Eigen::Vector3d> pt2)
{
  Eigen::Vector3d ecef1;
  Eigen::Vector3d ecef2;
  lla2ecef<_ref>(pt1, ecef1);
  lla2ecef<_ref>(pt2, ecef2);
  return (ecef1 - ecef2).norm();
}


/***************************************************************************
 * \brief Returns geodesic distance using Vincenty's formula. Throws a
 * runtime_error if it doesn't converge.
 ***************************************************************************/
template<typename _ref = WGS84>
double distGeodesicVincenty(
  const Eigen::Ref<const Eigen::Vector2d> pt1,
  const Eigen::Ref<const Eigen::Vector2d> pt2,
  const double tol = 1e-10,
  const uint64_t maxIter = 1000)
{
  double sin_sigma, cos_sigma, sigma, sin_alpha, cos_sq_alpha, cos2sigma;  // NOLINT
  double C, lam_pre;  // NOLINT

  // convert to radians
  const auto & latp = pt1[0];
  const auto & latc = pt2[0];
  const auto & longp = pt1[1];
  const auto & longc = pt2[1];

  const double u1 = atan((1 - _ref::f) * tan(latc));
  const double u2 = atan((1 - _ref::f) * tan(latp));

  const double lon = longp - longc;
  double lam = lon;
  double diff = 1.;
  std::size_t iter = 0;
  bool converged = false;
  while (iter < maxIter) {
    sin_sigma = sqrt(
      powFast<2>((cos(u2) * sin(lam))) +
      powFast<2>(cos(u1) * sin(u2) - sin(u1) * cos(u2) * cos(lam)));
    cos_sigma = sin(u1) * sin(u2) + cos(u1) * cos(u2) * cos(lam);
    sigma = atan(sin_sigma / cos_sigma);
    sin_alpha = (cos(u1) * cos(u2) * sin(lam)) / sin_sigma;
    cos_sq_alpha = 1. - powFast<2>(sin_alpha);
    cos2sigma = cos_sigma - ((2. * sin(u1) * sin(u2)) / cos_sq_alpha);
    C = (_ref::f / 16.) * cos_sq_alpha * (4 + _ref::f * (4. - 3. * cos_sq_alpha));
    lam_pre = lam;
    lam = lon + (1. - C) * _ref::f * sin_alpha *
      (sigma +
      C * sin_sigma * (cos2sigma + C * cos_sigma * (2. * powFast<2>(cos2sigma) - 1.)));
    diff = fabs(lam_pre - lam);

    if (fabs(diff) <= tol) {
      converged = true;
      break;
    }
    iter++;
  }

  if (!converged) {
    throw std::runtime_error("distGeodesicVincenty failed to converge.");
  }

  const double usq = cos_sq_alpha * ((_ref::a2 - _ref::b2) / _ref::b2);
  const double A = 1. + (usq / 16384.) * (4096. + usq * (-768. + usq * (320. - 175. * usq)));
  const double B = (usq / 1024.) * (256. + usq * (-128. + usq * (74. - 47. * usq)));
  const double delta_sig =
    B * sin_sigma *
    (cos2sigma + 0.25 * B *
    (cos_sigma * (-1. + 2. * powFast<2>(cos2sigma)) -
    (1. / 6.) * B * cos2sigma * (-3. + 4. * powFast<2>(sin_sigma)) *
    (-3. + 4. * powFast<2>(cos2sigma))));

  return _ref::b * A * (sigma - delta_sig);
}


/***************************************************************************
 * \brief Returns great circle distance using mean curvature radius
 * at the mean latitude of the points.
 ***************************************************************************/
template<typename _ref = WGS84>
double distGreatCircle(
  const Eigen::Ref<const Eigen::Vector2d> pt1,
  const Eigen::Ref<const Eigen::Vector2d> pt2)
{
  const auto & lat1 = pt1[0];
  const auto & lat2 = pt2[0];
  const auto & lon1 = pt1[1];
  const auto & lon2 = pt2[1];

  const double dLat = (lat2 - lat1);
  const double dLon = (lon2 - lon1);

  const double a = powFast<2>(sin(dLat / 2.)) + powFast<2>(sin(dLon / 2.)) * cos(lat1) * cos(lat2);
  const double c = 2. * asin(sqrt(a));

  return _ref::r * c;
}

}  // namespace cbr

#endif  // CBR_MATH__GEODETIC_HPP_
