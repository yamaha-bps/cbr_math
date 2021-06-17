// Copyright 2020 Yamaha Motor Corporation, USA
#ifndef CBR_MATH__EIGEN_TRAITS_HPP_
#define CBR_MATH__EIGEN_TRAITS_HPP_

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <functional>
#include <map>
#include <type_traits>
#include <utility>
#include <vector>

namespace cbr
{

/***************************************************************************
 * \brief Eigen type traits
 ***************************************************************************/
template<typename derived>
using is_eigen = std::is_base_of<Eigen::EigenBase<std::decay_t<derived>>, std::decay_t<derived>>;

template<typename T>
inline constexpr bool is_eigen_v = is_eigen<T>::value;

template<typename derived>
using is_eigen_dense = std::is_base_of<Eigen::DenseBase<std::decay_t<derived>>,
    std::decay_t<derived>>;
template<typename T>
inline constexpr bool is_eigen_dense_v = is_eigen_dense<T>::value;

template<typename derived>
using is_eigen_sparse = std::is_base_of<Eigen::SparseMatrixBase<std::decay_t<derived>>,
    std::decay_t<derived>>;
template<typename T>
inline constexpr bool is_eigen_sparse_v = is_eigen_sparse<T>::value;

template<typename derived>
using is_eigen_matrix = std::is_base_of<Eigen::MatrixBase<std::decay_t<derived>>,
    std::decay_t<derived>>;
template<typename T>
inline constexpr bool is_eigen_matrix_v = is_eigen_matrix<T>::value;

template<typename derived>
using is_eigen_array = std::is_base_of<Eigen::ArrayBase<std::decay_t<derived>>,
    std::decay_t<derived>>;
template<typename T>
inline constexpr bool is_eigen_array_v = is_eigen_array<T>::value;

/***************************************************************************
 * \brief Alias for aligned stl containers
 ***************************************************************************/
template<typename T>
using vector_aligned = std::vector<T, Eigen::aligned_allocator<T>>;

template<typename T1, typename T2, typename Compare = std::less<T1>>
using map_aligned = std::map<T1, T2, Compare, Eigen::aligned_allocator<std::pair<const T1, T2>>>;


}  // namespace cbr

#endif  // CBR_MATH__EIGEN_TRAITS_HPP_
