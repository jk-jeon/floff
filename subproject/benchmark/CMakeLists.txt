cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(benchmark LANGUAGES CXX)

set(benchmark_headers include/benchmark.h)

# ---- Add dependencies ----

include(FetchContent)
if (NOT TARGET floff)
    FetchContent_Declare(floff SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
    FetchContent_MakeAvailable(floff)
endif()
if (NOT TARGET common)
    FetchContent_Declare(common SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../common")
    FetchContent_MakeAvailable(common)
endif()
if (NOT TARGET ryu)
    FetchContent_Declare(ryu SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../3rdparty/ryu")
    FetchContent_MakeAvailable(ryu)
endif()
if (NOT TARGET fmt)
    FetchContent_Declare(fmt SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../3rdparty/fmt")
    FetchContent_MakeAvailable(fmt)
endif()

set(benchmark_sources
        source/floff.cpp
        source/ryu_printf.cpp
	source/fmt.cpp
        source/benchmark.cpp)

add_executable(benchmark ${benchmark_headers} ${benchmark_sources})

target_compile_features(benchmark PRIVATE cxx_std_17)

target_include_directories(benchmark
        PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

target_link_libraries(benchmark
        PRIVATE
        ryu::ryu
	fmt::fmt
        floff::common)

# ---- MSVC Specifics ----
if (MSVC)
    # No need to not generate PDB
    # /permissive- should be the default
    # The compilation will fail without /experimental:newLambdaProcessor
    # See also https://gitlab.kitware.com/cmake/cmake/-/issues/16478
    target_compile_options(benchmark PUBLIC
              /Zi /permissive-
              $<$<CONFIG:Release>:/GL>)
    target_link_options(benchmark PUBLIC /LTCG /DEBUG:FASTLINK)
    set_target_properties(benchmark PROPERTIES 
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
endif()