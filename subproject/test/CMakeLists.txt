cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(floff_tests LANGUAGES CXX)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# ---- Add dependencies ----

# We want warnings in tests
set(floff_INCLUDE_WITHOUT_SYSTEM ON CACHE INTERNAL "")

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

# ---- Tests ----

enable_testing()

# setup convenience umbrella targets
add_custom_target(floff-test-build)       # target for building all tests
add_custom_target(floff-test-run          # target for running all tests
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C $<CONFIG> ${FLOFF_CTEST_OPTIONS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS floff-test-build)
add_custom_target(floff-test-run-verbose  # target for running all tests in verbose mode
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C $<CONFIG> ${FLOFF_CTEST_OPTIONS} -VV
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS floff-test-build)
set_target_properties(floff-test-build PROPERTIES FOLDER floff/test)
set_target_properties(floff-test-run PROPERTIES FOLDER floff/test)
set_target_properties(floff-test-run-verbose PROPERTIES FOLDER floff/test)
if(NOT TARGET test)  # maybe this target is defined in a higher level project
  add_custom_target(test)
  set_target_properties(test PROPERTIES FOLDER floff/test)
endif()
add_dependencies(test floff-test-run)

function(floff_add_test NAME)
  cmake_parse_arguments(TEST "RYU" "" "" ${ARGN})
  set(floff floff::floff)
  add_executable(${NAME} source/${NAME}.cpp)
  add_dependencies(floff-test-build ${NAME})
  target_link_libraries(${NAME} PRIVATE ${floff} floff::common)
  if(TEST_RYU)
    target_link_libraries(${NAME} PRIVATE ryu::ryu)
  endif()
  target_compile_features(${NAME} PRIVATE cxx_std_17)
  add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${NAME}>)
  # ---- MSVC Specifics ----
  if (MSVC)
      # No need to not generate PDB
      # /permissive- should be the default
      # See also https://gitlab.kitware.com/cmake/cmake/-/issues/16478
      target_compile_options(${NAME} PUBLIC
              /Zi /permissive-
              $<$<CONFIG:Release>:/GL>)
      target_link_options(${NAME} PUBLIC $<$<CONFIG:Release>:/LTCG> /DEBUG:FASTLINK)
      set_target_properties(${NAME} PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}")
  endif()
  set_target_properties(${NAME} PROPERTIES FOLDER floff/test)
endfunction()

floff_add_test(float_exhaustive)
