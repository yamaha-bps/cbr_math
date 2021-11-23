cmake_minimum_required(VERSION 3.15)
project(cbr_math VERSION 0.1.0)

# ---------------------------------------------------------------------------------------
# GLOBAL FLAGS
# ---------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ---------------------------------------------------------------------------------------
# OPTIONS
# ---------------------------------------------------------------------------------------

include(CTest)

# Release type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# ---------------------------------------------------------------------------------------
# DEPENDENCIES
# ---------------------------------------------------------------------------------------

find_package(Eigen3 3.3 REQUIRED)
find_package(Boost REQUIRED)


# ---------------------------------------------------------------------------------------
# TARGETS
# ---------------------------------------------------------------------------------------

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME}
INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(${PROJECT_NAME}
INTERFACE
  Eigen3::Eigen
)


# ---------------------------------------------------------------------------------------
# INSTALLATION
# ---------------------------------------------------------------------------------------

include(GNUInstallDirs)

# Header files
install(
  DIRECTORY    ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}
  DESTINATION  ${CMAKE_INSTALL_INCLUDEDIR}
)

# Targets
install(
  TARGETS               ${PROJECT_NAME}
  EXPORT                ${PROJECT_NAME}_targets
  LIBRARY DESTINATION   ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION   ${CMAKE_INSTALL_LIBDIR}
)


# ---------------------------------------------------------------------------------------
# CMAKE CONFIGURATION
# ---------------------------------------------------------------------------------------

set(export_dest_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Generate TARGETS files
install(
  EXPORT       ${PROJECT_NAME}_targets
  FILE         ${PROJECT_NAME}Targets.cmake
  NAMESPACE    ${PROJECT_NAME}::
  DESTINATION  ${export_dest_dir}
)

# Generate CONFIG file
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/ConfigTemplate.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${export_dest_dir}
)

# Generate VERSION file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Version.cmake
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Install CONFIG and VERSION files
install(
  FILES        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
               ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Version.cmake
  DESTINATION  ${export_dest_dir}
)

# Custom uninstall target
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake
  IMMEDIATE @ONLY
)
add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake
)


# ---------------------------------------------------------------------------------------
# TESTING
# ---------------------------------------------------------------------------------------

if(BUILD_TESTING)
  include(GoogleTest)
  find_package(GTest REQUIRED)

  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Werror)

  if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    add_compile_options(-Wno-unused-local-typedef -Wno-mismatched-tags)
  endif()

  # Math
  add_executable(${PROJECT_NAME}_test_math test/test_math.cpp)
  target_link_libraries(${PROJECT_NAME}_test_math PRIVATE ${PROJECT_NAME} GTest::Main)
  gtest_discover_tests(${PROJECT_NAME}_test_math)

  # Geodetic
  add_executable(${PROJECT_NAME}_test_geodetic test/test_geodetic.cpp)
  target_link_libraries(${PROJECT_NAME}_test_geodetic PRIVATE ${PROJECT_NAME} GTest::Main)
  gtest_discover_tests(${PROJECT_NAME}_test_geodetic)

  # Interp
  add_executable(${PROJECT_NAME}_test_interp test/test_interp.cpp)
  target_link_libraries(${PROJECT_NAME}_test_interp PRIVATE ${PROJECT_NAME} GTest::Main)
  gtest_discover_tests(${PROJECT_NAME}_test_interp)
endif()

