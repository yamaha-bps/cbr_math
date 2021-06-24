// Copyright Yamaha 2021
// MIT License
// https://github.com/yamaha-bps/cbr_math/blob/master/LICENSE

#ifndef CBR_MATH__GEODETIC_DATA_HPP_
#define CBR_MATH__GEODETIC_DATA_HPP_

namespace cbr::geo
{

/***************************************************************************
 * Parameters of the WGS 84
 ***************************************************************************/
struct WGS84
{
  constexpr static double a{6378137.};            // semi-major axis
  constexpr static double f{1. / 298.257223563};  // flattening
  constexpr static double b{a * (1. - f)};        // semi-minor axis
  constexpr static double e2{f * (2. - f)};       // eccentricity squared
  constexpr static double a2{a * a};              // semi-minor axis
  constexpr static double b2{b * b};              // semi-minor axis
  constexpr static double r{(2. * a + b) / 3.};   // mean radius
};

}  // namespace cbr::geo

#endif  // CBR_MATH__GEODETIC_DATA_HPP_
