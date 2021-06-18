cmake_minimum_required(VERSION 3.15)
project(cbr_math VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_EXAMPLES "Build examples." OFF)
option(BUILD_TESTING "Build tests." OFF)
option(CODE_COVERAGE "Code coverage for tests." OFF)

find_package(Eigen3 3.3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(autodiff REQUIRED)

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
  Sophus::Sophus
  autodiff::autodiff
)

# ---------------------------------------------------------------------------------------
# EXAMPLES
# ---------------------------------------------------------------------------------------

if(BUILD_EXAMPLES)
  IF(EXISTS examples)
    add_subdirectory(examples)
  endif()
endif()

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

# ---------------------------------------------------------------------------------------
# TESTING
# ---------------------------------------------------------------------------------------

if(BUILD_TESTING)
  enable_testing()

  if (CODE_COVERAGE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(WARNING "Code coverage requires compiler to be either GCC or clang")
    set(CODE_COVERAGE OFF)
  endif()

  include(GoogleTest)
  find_package(GTest REQUIRED)

  # Math
  add_executable(${PROJECT_NAME}_test_math
    test/test_math.cpp
  )
  target_compile_options(${PROJECT_NAME}_test_math PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(${PROJECT_NAME}_test_math PRIVATE ${PROJECT_NAME} GTest::Main ${YAML_CPP_LIBRARIES})
  gtest_discover_tests(${PROJECT_NAME}_test_math)
  if (CODE_COVERAGE)
    target_compile_options(${PROJECT_NAME}_test_math PRIVATE -O0 -g --coverage -fno-inline -fno-inline-small-functions -fno-default-inline)
    target_link_libraries(${PROJECT_NAME}_test_math PRIVATE --coverage)
  endif()

  # Sophus
  add_executable(${PROJECT_NAME}_test_sophus
    test/test_sophus.cpp
  )
  target_compile_options(${PROJECT_NAME}_test_sophus PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(${PROJECT_NAME}_test_sophus PRIVATE ${PROJECT_NAME} GTest::Main ${YAML_CPP_LIBRARIES})
  gtest_discover_tests(${PROJECT_NAME}_test_sophus)
  if (CODE_COVERAGE)
    target_compile_options(${PROJECT_NAME}_test_sophus PRIVATE -O0 -g --coverage -fno-inline -fno-inline-small-functions -fno-default-inline)
    target_link_libraries(${PROJECT_NAME}_test_sophus PRIVATE --coverage)
  endif()

  # Geodetic
  add_executable(${PROJECT_NAME}_test_geodetic
    test/test_geodetic.cpp
  )
  target_compile_options(${PROJECT_NAME}_test_geodetic PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(${PROJECT_NAME}_test_geodetic PRIVATE ${PROJECT_NAME} GTest::Main ${YAML_CPP_LIBRARIES})
  gtest_discover_tests(${PROJECT_NAME}_test_geodetic)
  if (CODE_COVERAGE)
    target_compile_options(${PROJECT_NAME}_test_geodetic PRIVATE -O0 -g --coverage -fno-inline -fno-inline-small-functions -fno-default-inline)
    target_link_libraries(${PROJECT_NAME}_test_geodetic PRIVATE --coverage)
  endif()

  # Interp
  add_executable(${PROJECT_NAME}_test_interp
    test/test_interp.cpp
  )
  target_compile_options(${PROJECT_NAME}_test_interp PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(${PROJECT_NAME}_test_interp PRIVATE ${PROJECT_NAME} GTest::Main ${YAML_CPP_LIBRARIES})
  gtest_discover_tests(${PROJECT_NAME}_test_interp)
  if (CODE_COVERAGE)
    target_compile_options(${PROJECT_NAME}_test_interp PRIVATE -O0 -g --coverage -fno-inline -fno-inline-small-functions -fno-default-inline)
    target_link_libraries(${PROJECT_NAME}_test_interp PRIVATE --coverage)
  endif()

  # Lie
  add_executable(${PROJECT_NAME}_test_lie
    test/lie/test_lie_bspline_autodiff.cpp
    test/lie/test_lie_bspline.cpp
    test/lie/test_lie.cpp
  )
  target_compile_options(${PROJECT_NAME}_test_lie PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(${PROJECT_NAME}_test_lie PRIVATE ${PROJECT_NAME} GTest::Main ${YAML_CPP_LIBRARIES})
  gtest_discover_tests(${PROJECT_NAME}_test_lie)
  if (CODE_COVERAGE)
    target_compile_options(${PROJECT_NAME}_test_lie PRIVATE -O0 -g --coverage -fno-inline -fno-inline-small-functions -fno-default-inline)
    target_link_libraries(${PROJECT_NAME}_test_lie PRIVATE --coverage)
  endif()


  # -------------------------------------------------------------------------------------
  # TEST COVERAGE
  # -------------------------------------------------------------------------------------

  if(CODE_COVERAGE)
    add_custom_target(testcov
      COMMAND lcov -c --include ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/\\* -d ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles -o ${PROJECT_NAME}_testcov.info
      COMMAND genhtml ${PROJECT_NAME}_testcov.info -o out
      COMMAND echo "Open file://${CMAKE_CURRENT_BINARY_DIR}/out/index.html in a browser to view results"
    )
  endif()
endif()
