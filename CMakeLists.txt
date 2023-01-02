cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(floff
        VERSION 0.0.1
        LANGUAGES CXX)

# ---- Includes ----

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# ---- Warning guard ----

# Protect dependents from this project's warnings if the guard isn't disabled
set(floff_warning_guard "SYSTEM")
if(floff_INCLUDE_WITHOUT_SYSTEM)
  set(floff_warning_guard "")
endif()

# ---- Declare library (floff) ----

add_library(floff INTERFACE)
add_library(floff::floff ALIAS floff)

set(floff_headers include/floff/floff.h)

target_include_directories(floff
        ${floff_warning_guard}
        INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")

target_compile_features(floff INTERFACE cxx_std_17)

# ---- Install ----

set(floff_directory "floff-${PROJECT_VERSION}")
set(floff_include_directory "${CMAKE_INSTALL_INCLUDEDIR}/${floff_directory}")
set(floff_install_targets "floff")

install(TARGETS ${floff_install_targets}
        EXPORT floffTargets
        ARCHIVE #
        DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        COMPONENT floff_Development
        INCLUDES #
        DESTINATION "${floff_include_directory}")

set(floff_install_cmakedir "${CMAKE_INSTALL_LIBDIR}/cmake/${floff_directory}")

write_basic_package_version_file(
        floffConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT)

install(EXPORT floffTargets
        NAMESPACE floff::
        DESTINATION "${floff_install_cmakedir}")

install(FILES
        "${PROJECT_SOURCE_DIR}/cmake/floffConfig.cmake"
        "${PROJECT_BINARY_DIR}/floffConfigVersion.cmake"
        DESTINATION "${floff_install_cmakedir}")

install(FILES ${floff_headers}
        DESTINATION "${floff_include_directory}/floff")

# ---- Subproject ----

option(floff_ENABLE_SUBPROJECT "Build subprojects as well" OFF)

if (floff_ENABLE_SUBPROJECT)
    add_subdirectory("subproject/benchmark")
    add_subdirectory("subproject/meta")
    add_subdirectory("subproject/test")
endif()

# ---- MSVC Specifics ----
if (MSVC)
    # No need to not generate PDB
    # /permissive- should be the default
    target_compile_options(floff INTERFACE
              /Zi /permissive-)
endif()