cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(floff_common LANGUAGES CXX)

include(FetchContent)
if (NOT TARGET floff)
    FetchContent_Declare(floff SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
    FetchContent_MakeAvailable(floff)
endif()

set(floff_common_headers
        include/big_uint.h
	include/continued_fractions.h
	include/rational_continued_fractions.h
	include/best_rational_approx.h
	include/good_rational_approx.h
        include/random_float.h)

set(floff_common_sources source/big_uint.cpp)

add_library(floff_common STATIC
        ${floff_common_headers}
        ${floff_common_sources})
add_library(floff::common ALIAS floff_common)

target_include_directories(floff_common
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_compile_features(floff_common PUBLIC cxx_std_17)

target_link_libraries(floff_common PUBLIC floff::floff)

# ---- MSVC Specifics ----
if (MSVC)
    # No need to not generate PDB
    # /permissive- should be the default
    target_compile_options(floff_common PUBLIC
              /Zi /permissive-)
endif()