cmake_minimum_required(VERSION 3.24)
project(OOP_Labs CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

# ---------- core library ----------
add_library(core STATIC
    src/Dataset.cpp
    src/Matrix.cpp
    src/DecisionStump.cpp
    src/SplitFinder.cpp
    src/CartRegressor.cpp
    src/CartClassifier.cpp
)
target_include_directories(core PUBLIC "${CMAKE_SOURCE_DIR}/include")

# ---------- console apps ----------
add_executable(Lab1 src/Lab1.cpp)
target_link_libraries(Lab1 PRIVATE core)

add_executable(Lab2_star src/Lab2.cpp)
target_link_libraries(Lab2_star PRIVATE core)

add_executable(Lab3 src/Lab3.cpp)
target_link_libraries(Lab3 PRIVATE core)

# ---------- tests ----------
enable_testing()

set(TESTS
    binary_test
    column_type_test
    encoding_test
    missing_value_test
    quantile
    consistency_test
    stump_test
    matrix_slice_test
    split_finder_test
    cart_regressor_test
    cart_classifier_test
)

foreach(test_name IN LISTS TESTS)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE core)
    add_test(NAME ${test_name}
             COMMAND ${test_name}
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endforeach()